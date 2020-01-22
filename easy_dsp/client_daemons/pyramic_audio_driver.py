"""
Copyright (C) 2020 Robin Scheibler (fakufaku@gmail.com)

This is an audio driver for the pyramic device connected via the network.

The driver relies on the availability of a virtual audio device such as
Soundflower on macOS or snd-aloop on Linux. It gets the samples from the network
via the easy-dsp websocket and routes them to the virtual audio interface.
"""
import argparse
from threading import Thread
from queue import Queue
import socket

from ws4py.client.threadedclient import WebSocketClient
import sounddevice as sd
import json
import time
import numpy as np


SOUNDFLOWER_2CH = "Soundflower (64ch)"


def select_device_by_name(name):

    found = False
    for dev_id, dev in enumerate(sd.query_devices()):
        if dev["name"].startswith(name):
            found = True
            break

    if not found:
        raise ValueError(f"Device {name} could not be found")

    else:
        # select as default device
        sd.default.device = dev_id

    return dev


# Connection with WSAudio
class StreamClient(WebSocketClient):
    def __init__(self, frame_queue, config_queue, *args, **kwargs):

        # derived class attributes
        self.frame_queue = frame_queue
        self.config_queue = config_queue
        self.buffer = None
        self.channels = 1

        # parent constructor
        super().__init__(*args, **kwargs)

    def received_message(self, m):

        if not m.is_binary:  # new configuration

            m = json.loads(m.data.decode("utf-8"))
            self.rate = m["rate"]
            self.channels = m["channels"]
            self.buffer_frames = m["buffer_frames"]
            self.volume = m["volume"]

            self.buffer = np.zeros((self.buffer_frames, self.channels), dtype=np.int16)
            self.config_queue.put(
                (self.buffer_frames, self.rate, self.channels, self.volume)
            )

        else:  # new audio data

            # We convert the binary stream into a 2D Numpy array of 16-bits integers
            raw_data = np.frombuffer(m.data, dtype=np.int16)
            audio_buffer = raw_data.reshape(-1, self.channels)
            if not self.frame_queue.full():
                self.frame_queue.put(audio_buffer)
            else:
                print("Buffer overflow")


# Change the configuration (WSConfig)
def change_config(rate=None, channels=None, buffer_frames=None, volume=None):
    if rate is None:
        rate = globals()["rate"]
    if channels is None:
        channels = globals()["channels"]
    if buffer_frames is None:
        buffer_frames = globals()["buffer_frames"]
    if volume is None:
        volume = globals()["volume"]

    class WSConfigClient(WebSocketClient):
        def opened(self):
            self.send(
                json.dumps(
                    {
                        "rate": rate,
                        "channels": channels,
                        "buffer_frames": buffer_frames,
                        "volume": volume,
                    }
                )
            )
            self.close()

    change_config_q = WSConfigClient(
        "ws://" + self.url + ":7322/", protocols=["http-only", "chat"]
    )
    change_config_q.connect()


class PyramicAudioDriver(object):
    def __init__(
        self,
        audio_device=None,
        url=None,
        rate=48000,
        channels=48,
        buffer_frames=None,
        volume=None,
        q_maxsize=10,
    ):
        self.audio_device = audio_device
        self.url = url
        self.rate = rate
        self.channels = channels
        self.buffer_frames = buffer_frames
        self.volume = volume

        self.q_maxsize = q_maxsize
        self.frame_queue = Queue(maxsize=q_maxsize)
        self.config_queue = Queue(maxsize=q_maxsize)

        self.current_buffer = None
        self.current_buffer_ptr = 0

        if self.audio_device is None:
            self.audio_device = SOUNDFLOWER_2CH

        if self.url is None:
            self.url = "192.168.2.26"

        # placeholder for the websocket client object
        self.ws = None

        self.dev_info = select_device_by_name(self.audio_device)
        sd.default.dtype = np.int16

        # We know that pyramic is
        sd.default.samplerate = self.rate
        time.sleep(1.0)

    def run(self):

        self.ws = StreamClient(
            self.frame_queue,
            self.config_queue,
            "ws://" + self.url + ":7321/",
            protocols=["http-only", "chat"],
        )

        self.clientThread = Thread(target=self._websocket_client_thread)
        self.clientThread.daemon = True
        self.clientThread.start()

        n_chan = self.dev_info["max_output_channels"]

        with sd.Stream(channels=n_chan, callback=self._audio_callback):
            try:
                while True:
                    if not self.config_queue.empty():
                        (
                            self.buffer_frames,
                            self.rate,
                            self.channels,
                            self.volume,
                        ) = self.config_queue.get()
                        self.print_config()

                    q_size = self.frame_queue.qsize()
                    if q_size > self.q_maxsize // 2:
                        print("Warning: queue is half full {q_size}/{self.q_maxsize}.")
                    sd.sleep(1000)

            except KeyboardInterrupt:
                print("Stopping the audio device")
                pass

    def _websocket_client_thread(self):

        try:
            self.ws.connect()
            self.ws.run_forever()
        except KeyboardInterrupt:
            self._stop()

    def _stop(self):
        print("Manual interuption.")
        if self.ws is not None:
            self.ws.close()

    def _audio_callback(self, indata, outdata, frames, t, status):
        if status:
            print(status)

        fresh = self.pop(outdata.shape[0])

        if fresh is not None:
            n_chan = np.minimum(outdata.shape[1], fresh.shape[1])
            outdata[:, :n_chan] = fresh[:, :n_chan]
        else:
            outdata[:] = 0
            print("Underflow")

    def print_config(self):
        print(
            f"buffer_frames: {self.buffer_frames} rate: {self.rate} "
            f"channels: {self.channels} volume: {self.volume}"
        )

    def pop(self, n_samples):

        if self.current_buffer is None:

            if self.frame_queue.empty():
                return None
            else:
                self.current_buffer = self.frame_queue.get()
                self.current_buffer_ptr = 0

        ptr = self.current_buffer_ptr
        buf = self.current_buffer

        if len(buf) - ptr >= n_samples:
            # the buffer chunk we need
            ret = buf[ptr : ptr + n_samples, :]
            self.current_buffer_ptr += n_samples

            # handle end of buffer
            if ptr == len(buf):
                self.current_buffer_ptr = 0
                self.current_buffer = None

            return ret

        else:

            # get the end of the current buffer
            ret_part1 = buf[ptr:, :]
            self.current_buffer_ptr = 0
            self.current_buffer = None

            # do a recursion to get new data
            ret_part2 = self.pop(n_samples - ret_part1.shape[0])

            # concatenate the data
            if ret_part2 is not None:
                ret = np.concatenate((ret_part1, ret_part2), axis=0)

            # pad with zeros if not enough samples are available
            if ret.shape[0] < n_samples:
                ret = np.concatenate(
                    (
                        ret,
                        np.zeros(
                            (n_samples - ret.shape[0], ret.shape[1]), dtype=ret.dtype
                        ),
                    ),
                    axis=0,
                )

            return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio driver for the Pyramic array")
    parser.add_argument(
        "--url",
        type=str,
        default="pyramic.local",
        help="The address of the Pyramic array on the network",
    )
    args = parser.parse_args()

    # convert the name to an IP address
    addr = socket.gethostbyname(args.url)

    PyramicAudioDriver(url=addr, q_maxsize=10).run()
