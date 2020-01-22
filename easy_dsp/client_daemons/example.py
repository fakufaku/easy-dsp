# ######################################################################################################################
# example.py
# ==========
# Minimum building block to get the audio stream from a Pyramic
#
# Author        : Sepand KASHANI [sep@zurich.ibm.com]
# ######################################################################################################################

import numpy
from easy_dsp.client_daemons import streaming as stream
import datetime

### Pyramic Configuration: these must be kept in sync with browser-config.h ############################################
EASY_DSP_VOLUME = 100
EASY_DSP_NUM_CHANNELS = 48
EASY_DSP_AUDIO_FREQ_HZ = 48000
EASY_DSP_AUDIO_FORMAT_BITS = 16
EASY_DSP_AUDIO_FORMAT_BYTES = EASY_DSP_AUDIO_FORMAT_BITS // 8
EASY_DSP_AUDIO_BUFFER_LENGTH_MS = 200
EASY_DSP_AUDIO_BUFFER_SIZE_BYTES = int((EASY_DSP_NUM_CHANNELS * EASY_DSP_AUDIO_FREQ_HZ * EASY_DSP_AUDIO_FORMAT_BYTES * \
                                        (EASY_DSP_AUDIO_BUFFER_LENGTH_MS / 1000.0)))
########################################################################################################################

### streaming.py Settings ##############################################################################################
#stream.EASY_DSP_BOARD_IP_ADDRESS = '10.42.0.2'
stream.EASY_DSP_BOARD_IP_ADDRESS = '192.168.2.26'
stream.EASY_DSP_WSAUDIO_SERVER_PORT = 7321
stream.EASY_DSP_WSCONFIG_SERVER_PORT = 7322
stream.sample_rate = EASY_DSP_AUDIO_FREQ_HZ
stream.channel_count = EASY_DSP_NUM_CHANNELS
stream.frame_count = EASY_DSP_AUDIO_BUFFER_SIZE_BYTES // (EASY_DSP_NUM_CHANNELS * EASY_DSP_AUDIO_FORMAT_BYTES)
stream.volume = EASY_DSP_VOLUME
########################################################################################################################

### Define Callbacks ###################################################################################################
def handle_samples(buffer):
    printProgress("handle_buffer: received {count} bytes | shape {shape} | type {dtype} | energy {pwr}".format(
                                                                                                count=buffer.nbytes,
                                                                                                shape=buffer.shape,
                                                                                                dtype=buffer.dtype,
                                                                                                pwr=numpy.mean(buffer**2)
                                                                                                ))

def handle_config(args=None):
    printProgress(
        "handle_config: new config ({frames},{sampleRate},{channelCount},{volume})".format(frames=stream.frame_count,
                                                                                           sampleRate=stream.sample_rate,
                                                                                           channelCount=stream.channel_count,
                                                                                           volume=stream.volume))

def printProgress(extraInfo=None):
   """
   Print to sys.stdout the name of the calling function along with a timestamp.
   If extraInfo is given, then replace the calling function name with extraInfo.

   :param extraInfo: string of user information
   """
   dt = datetime.datetime.now()
   if extraInfo:
       info = '{year:=04d}-{month:=02d}-{day:=02d}/{hour:02d}:{minute:=02d}:{second:=02d}  {info}'.format(
           year=dt.year, month=dt.month, day=dt.day,
           hour=dt.hour, minute=dt.minute, second=dt.second,
           info=extraInfo
       )
   else:
       info = '{year:=04d}-{month:=02d}-{day:=02d}/{hour:02d}:{minute:=02d}:{second:=02d}  {function}'.format(
           year=dt.year, month=dt.month, day=dt.day,
           hour=dt.hour, minute=dt.minute, second=dt.second,
           function=sys._getframe(1).f_code.co_name
       )
   print(info)

########################################################################################################################

if __name__ == '__main__':
    stream.change_config()
    stream.process_samples = handle_samples
    stream.process_config = handle_config
    stream.start()
    stream.loop_callbacks()
