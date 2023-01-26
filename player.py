from fbuffer import Buffer
from detector import Detector

# For multithreading
from threading import Thread, Lock, Condition
from time import perf_counter, sleep

# For input arguments
#from sys import argv
import argparse

# For img processing 
import numpy as np
import cv2

# For audio processing
from scipy.io import wavfile
import simpleaudio as sa

# Another File
from shot_detection import ShotDetector
from detector import Detector

# Globals:
global IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH
global SAMPLING_RATE, AUDIO_CHANNELS, AUDIO_BYTES_PER_SAMPLE

# Create buffer and synchronization global variables
global img_buffer, img_buffer_lock, img_buffer_cv
global didQuit, restart, didFinishReading
didQuit = False
restart = False
didFinishReading = False
framesInFile = 0
img_buffer = Buffer(10) # 10 is the number of seconds we store of the video at any one time
img_buffer_lock = Lock()
img_buffer_cv = Condition(img_buffer_lock)
    
class MediaPlayer():
    def __init__(self, videoFile):
        global SAMPLING_RATE, AUDIO_CHANNELS, AUDIO_BYTES_PER_SAMPLE
        global img_buffer, img_buffer_cv
        global audio_buffer

        self.videoFile = videoFile
        #self.init_img = []

    def init_play(self):
        print("[INFO] Controls: P - Play/Pause, S - Stop and Reset, Q - Quit")
        global restart
        # Keep playing video until video player is quit
        playAgain = True # Will play again if true
        while(playAgain):
            img_buffer_cv.acquire()
            while (img_buffer.__len__() == 0 or SAMPLING_RATE == 0 or restart):
                img_buffer_cv.notify()
                img_buffer_cv.wait()
            self.second_of_frames = img_buffer.pop()
            self.init_img = self.second_of_frames[0, :, :, :]
            img_buffer_cv.release()

            self.frame_num = 0 # Only goes to 30
            self.start_time = perf_counter()
            self.time_to_sleep = 10
            self.pause = True # Starting the video on first frame, paused
            self.audio_object = None
            
            self.totalFrames = 0
            self.timeBreak = 0
            self.pause_time_difference = 0

            playAgain = self.play()
            if playAgain: restart = True

    def play(self):
        global img_buffer, img_buffer_cv
        global audio_buffer
        global didQuit, didFinishReading

        window = self.videoFile
        self.start_time = perf_counter()
        self.pause_time_difference = perf_counter()
        thisFrame = self.init_img
        keepPlaying = True
        while (not didFinishReading or img_buffer.__len__() > 0 or self.frame_num < 29):

            if not self.pause:
                img_buffer_cv.acquire()
                while (img_buffer.__len__() == 0 and not didFinishReading):
                    img_buffer_cv.wait()

                thisFrame = self.get_frame()
                self.frame_num += 1

                img_buffer_cv.notify()
                img_buffer_cv.release()

                cv2.imshow(window, cv2.cvtColor(thisFrame, cv2.COLOR_BGR2RGB))
                keyPress = cv2.waitKey(self.time_to_sleep)
                if keyPress == ord('p'): # PAUSE / PLAY
                    self.pause = True
                    self.pause_save_state()
                elif keyPress == ord('q'): # QUIT
                    didQuit = True
                    keepPlaying = False
                    break
                elif keyPress == ord('s'): # STOP
                    if (self.audio_object.is_playing()):
                        self.audio_object.stop()
                    break

                shift = perf_counter() - (self.start_time+self.timeBreak + ((1.0/30.0)*self.totalFrames))
                sleep(max(.03333 - shift, 0))
                    
            else: # Paused, stick on frame
                cv2.imshow(window, cv2.cvtColor(thisFrame, cv2.COLOR_BGR2RGB))

                # UNPAUSED
                keyPress = cv2.waitKey(0) 
                if keyPress == ord('p'): # PLAY / PAUSE
                    self.pause = False
                    self.audio_object = sa.play_buffer(audio_buffer[self.totalFrames*samples_per_frame:len(audio_buffer)], AUDIO_CHANNELS, AUDIO_BYTES_PER_SAMPLE, SAMPLING_RATE)
                    self.timeBreak += perf_counter() - self.pause_time_difference
                elif keyPress == ord('q'): # QUIT
                    didQuit = True
                    keepPlaying = False
                    break
                elif keyPress == ord('s'): # STOP
                    if (self.audio_object.is_playing()):
                        self.audio_object.stop()
                    break
            

        # Quit, we're finished with media playing
        img_buffer_cv.acquire()

        if didFinishReading and img_buffer.__len__() == 0:
            keepPlaying = False
            didQuit = True

        img_buffer_cv.notify()
        img_buffer_cv.release()
        cv2.destroyAllWindows()
        
        return keepPlaying
            
    
    def get_frame(self):
        if self.frame_num == 30:
            self.second_of_frames = img_buffer.pop()
            self.frame_num = 0
        self.totalFrames += 1
        return self.second_of_frames[min(self.frame_num, 29)] # In the worst case we dont crash, but display the last frame

    def pause_save_state(self):
        print("Debug: Pausing video...")
        self.audio_object.stop()
        self.pause_time_difference = perf_counter()


# Thread to process rgb file for video during playback
def img_proc_thread(fpath, buf_max_sec=10):
    # Declare globals
    global IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH
    global FRAMES_PER_SECOND
    global didQuit, restart, didFinishReading

    # Define constants
    VID_PATH = fpath
    IMG_WIDTH  = 480
    IMG_HEIGHT = 270
    IMG_DEPTH  = 3
    FRAMES_PER_SECOND = 30
    IMG_AREA = IMG_WIDTH * IMG_HEIGHT

    vid_f = open(VID_PATH, 'rb')
    vid_f_size = IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH

    # Image reader loop
    while True:
        img_buffer_cv.acquire()

        # User pressed quit
        if (didQuit): 
            img_buffer_cv.notify()
            img_buffer_cv.release()
            vid_f.close()
            return

        # BUFFER IS FULL RIGHT NOW
        while (img_buffer.__len__() > buf_max_sec and not restart):
            if (didQuit):
                vid_f.close()
                img_buffer_cv.notify()
                img_buffer_cv.release()
                return
            img_buffer_cv.wait()
        '''
        if (restart):
            vid_f.close()
            vid_f = open(VID_PATH, 'rb')
            img_buffer.clear()
            restart = False
            didFinishReading = False
        '''
        oneSecond = np.zeros((FRAMES_PER_SECOND, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), dtype=np.uint8)
        for f in range(0, FRAMES_PER_SECOND):
            bit_str = vid_f.read(vid_f_size)

            # ENTIRE FILE HAS BEEN READ
            if not bit_str or len(bit_str) < vid_f_size:
                print("finished reading file")
                img_buffer_cv.notify()
                img_buffer_cv.release()
                vid_f.close()
                didFinishReading = True
                return

            bit_arr = np.frombuffer(bit_str, np.uint8)
            
            for d in range(IMG_DEPTH):
                channel = np.reshape(bit_arr[d * IMG_AREA:(d+1) * IMG_AREA], 
                                    (IMG_HEIGHT, IMG_WIDTH))
                oneSecond[f, :,:,d] = channel

        img_buffer.add(oneSecond)

        if (restart):
            print('Debug: restarted')
            vid_f.close()
            vid_f = open(VID_PATH, 'rb')
            img_buffer.clear()
            restart = False
            didFinishReading = False

        img_buffer_cv.notify()
        img_buffer_cv.release()

# Thread to process wav file for audio during playback
def wav_proc_thread(fpath):
    # Declare globals
    global AUDIO_CHANNELS, AUDIO_BYTES_PER_SAMPLE, SAMPLING_RATE
    global samples_per_frame, audio_buffer

    # Define constants
    WAV_PATH = fpath
    AUDIO_CHANNELS = 1
    AUDIO_BYTES_PER_SAMPLE = 2
    SAMPLING_RATE = 48000
    samples_per_frame = 1600

    # Read audio into buffer  
    SAMPLING_RATE, audio_buffer = wavfile.read(WAV_PATH)


def main(argc, argv):
    # inputVideo = args[1] # video file to read input from
    # inputAudio = args[2] # audio file to read input from
    inputVideo = argv['video']
    inputAudio = argv['audio']
    shot_detector = ShotDetector(inputVideo, inputAudio)

    # Mihai --- My code is going to go in here, we don't need the 3rd if/else if anymore
    if argv['video_out'] and argv['audio_out']:
        outputVideo = argv['video_out']
        outputAudio = argv['audio_out']
        shot_detector.vid_f_out = outputVideo
        shot_detector.aud_f_out = outputAudio

        logo_detector = Detector(argv['model'], argv['config']) if argv['model'] and argv['config'] else None
        shot_detector.logo_detector = logo_detector

        print("[INFO] beginning ad search...")
        shots = shot_detector.ad_search()

        #shot_detector.write_output(shots)
        vid_bit_arr = shot_detector.process_logos()
        shot_detector.shot_logo_sync(shots)
        for shot in shots:
            print(shot)
        ads_bit_arr = shot_detector.process_ads(shots, vid_bit_arr)
        shot_detector.write_output(ads_bit_arr)
        inputVideo = outputVideo
        inputAudio = outputAudio

    """
    if len(args) == 5: # We want to remove ads and then play the videos
        outputVideo = args[3] # video file to output new rgb to
        outputAudio = args[4] # audio file to output new wav to
        # ad_search = Thread(target='''ad search function''', args=(inputVideo)) # Updates globals which hold the ad start frames and end frames
        shots = ad_search(inputVideo)

        #----DHWANIL's FUNCTION CALL TO REMOVE ADS AND OUTPUT VIDEO/AUDIO TO NEW FILES-----
        write_output(shots, inputVideo, inputAudio, outputVideo, outputAudio)
        inputVideo = outputVideo
        inputAudio = outputAudio
    """

    """
    elif argv['targets'] is not None:
        assert argv['video_out']
        assert argc['video_out']

        outputVideo = argv['video_out']
        outputAudio = argv['audio_out']
        logo_search = Thread(target='''logo search function''', args=(inputVideo,)) # Updates globals with the specfic logos found if any
    """

    """
    elif len(args) == 6: # We want to logo detect too!
        outputVideo = args[3] # video file to output new rgb to
        outputAudio = args[4] # audio file to output new wav to
        # ad_search = Thread(target='''ad search function''', args=(inputVideo,)) # Updates globals which hold the ad start frames and end frames
        logo_search = Thread(target='''logo search function''', args=(inputVideo,)) # Updates globals with the specfic logos found if any
        '''
            ad_search and logo search probably need to access the same video file (buffer of frames) while they work
            we should probably make synchronization primitives to handle this
            we can either read the whole thing into memory and get rid of it when were done processing or...
                build the functions in a way that the section being analyzed isnt thrown away until both threads have processed it for their tasks
        '''
    """
        #----FUNCTION CALL TO REPLACE ADS BASED ON FOUND LOGOS AND OUTPUT VIDEO/AUDIO TO NEW FILES-----


    image_thread = Thread(target=img_proc_thread, args=(inputVideo,))
    image_thread.start()
    audio_thread = Thread(target=wav_proc_thread, args=(inputAudio,))
    audio_thread.start()

    app = MediaPlayer(inputVideo)
    app.init_play()

    # Join threads
    image_thread.join()
    audio_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs video player with various input arguments")
    parser.add_argument('-v', '--video', required=True,
                        help="Path to input .rgb file")
    parser.add_argument('-a', '--audio', required=True,
                        help="Path to input .wav file")
    parser.add_argument('--video_out', default=None,
                        help="Optional path to output .rgb file")
    parser.add_argument('--audio_out', default=None,
                        help="Optional path to output .wav file")
    parser.add_argument('-t', '--target', default=None,
                        help='Optional path to target ads (WARN: Currently not used)')
    parser.add_argument('-m', '--model', default=None,
                        help='Optional path to pytorch model .pkl file')
    parser.add_argument('-c', '--config', default=None,
                        help="Config path to .pkl file to use for pytorch model")
    argv = vars(parser.parse_args())
    main(len(argv), argv)



    """
    if (len(argv) != 3 and len(argv) != 5 and len(argv) != 6):
        print ('Invalid arguments. Usage information below for baseline (1), removed ads (2), and targeted ads (3)')
        print ('1: python3 player.py <input rgb> <input wav>')
        print ('2: python3 player.py <input rgb> <input wav> <output rgb> <output wav>')
        print ('3: python3 player.py <input rgb> <input wav> <output rgb> <output wav> <-target>')
    else:
        main(argv)
    """
