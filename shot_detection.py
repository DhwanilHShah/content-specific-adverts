import os

from scipy.io import wavfile
import numpy as np
import simpleaudio as sa
import cv2
from tqdm import tqdm
import glob


class ShotDetector:
    IMG_WIDTH = 480
    IMG_HEIGHT = 270
    IMG_DEPTH = 3
    IMG_AREA = IMG_WIDTH * IMG_HEIGHT
    VID_F_SIZE = IMG_AREA * IMG_DEPTH
    FRAME_RATE = 30  # frames per sec

    def __init__(self, vid_f_in, aud_f_in, vid_f_out=None, aud_f_out=None, logo_detector=None):
        self.vid_f_in = vid_f_in
        self.aud_f_in = aud_f_in
        self.vid_f_out = vid_f_out
        self.aud_f_out = aud_f_out
        self.logo_detector = logo_detector
        self.label_to_logo_map = {'Starbucks': 'sbux', 'Subway': 'subway', 'ae': 'ae', 'hrc': 'hrc',
                                  'mcd': 'mcd', 'nfl': 'nfl'}

    def frame_from_bit_arr(self, bit_arr):
        img = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_DEPTH), dtype=np.uint8)

        for d in range(self.IMG_DEPTH):
            channel = np.reshape(bit_arr[d * self.IMG_AREA : (d + 1) * self.IMG_AREA],
                                 (self.IMG_HEIGHT, self.IMG_WIDTH))
            img[:, :, self.IMG_DEPTH - 1 - d] = channel

        return img

    def bit_arr_from_frame(self, frame):
        bit_arr = np.zeros((self.VID_F_SIZE,), dtype=np.uint8)
        for d in range(self.IMG_DEPTH):
            bit_arr[d * self.IMG_AREA: (d + 1) * self.IMG_AREA] = frame[:, :, self.IMG_DEPTH - 1 - d].flatten()

        return bit_arr

    def frame_extraction(self, vid_f):
        bit_str = vid_f.read(self.VID_F_SIZE)
        if not bit_str or len(bit_str) < self.VID_F_SIZE:
            print("finished reading file, exiting")
            return []

        bit_arr = np.frombuffer(bit_str, np.uint8)
        img = self.frame_from_bit_arr(bit_arr)
        '''
        img = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_DEPTH), dtype=np.uint8)

        for d in range(self.IMG_DEPTH):
            channel = np.reshape(bit_arr[d * self.IMG_AREA:(d + 1) * self.IMG_AREA],
                                 (self.IMG_HEIGHT, self.IMG_WIDTH))
            img[:, :, (2 - d)] = channel
        '''
        return img

    def absolute_diff(self, p_img, img):
        diff = cv2.absdiff(img, p_img)
        diff_sum = np.sum(diff)
        diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
        return diff_sum_mean

    def create_histogram(self, img):
        h_bins = 50
        s_bins = 60
        histSize = [h_bins, s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges  # concat lists
        # Use the 0-th and 1-st channels
        channels = [0, 1]

        hist_base = cv2.calcHist([img], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return hist_base

    def ad_search(self):
        vid_f = open(self.vid_f_in, 'rb')

        frame_count = 1
        p_img = self.frame_extraction(vid_f)
        p_img_h = self.create_histogram(p_img)

        shot_count = 1
        shot = [shot_count, frame_count]

        result = []

        while True:
            img = self.frame_extraction(vid_f)
            frame_count += 1

            if len(img) == 0:
                break

            img_h = self.create_histogram(img)

            diff = self.absolute_diff(p_img, img)
            compHist = cv2.compareHist(img_h, p_img_h, 0)

            if compHist < 0.5 or diff > 200 or (compHist < 0.87 and diff > 135):
                shot.append(frame_count - 1)
                shot.append(compHist)
                shot.append(diff)

                if shot[2] - shot[1] > 299 or (len(result) > 0 and result[-1][-1] == 1 and result[-1][3] > 0.6):
                    shot.append(1)
                    if len(result) > 1 and result[-1][-1] == 0 and result[-1][3] > 0.75:
                        result[-1][-1] = 1
                else:
                    shot.append(0)

                result.append(shot)

                shot_count += 1

                shot = [shot_count, frame_count]

            p_img = img
            p_img_h = img_h

        shot.append(frame_count - 1)
        shot.append(-1)
        shot.append(-1)

        if shot[2] - shot[1] > 299 or (result[-1][-1] == 1 and result[-1][3] > 0.55):
            shot.append(1)
            if len(result) > 1 and result[-1][-1] == 0 and result[-1][3] > 0.75 and result[-1][4] < 200:
                result[-1][-1] = 1
        else:
            shot.append(0)

        result.append(shot)

        return result

    def process_ads(self, result, vid_bit_arr):
        # output_video_file = open(self.vid_f_out, "wb")
        '''
        video = open(self.vid_f_in, 'rb')
        vid_bit_str = video.read()
        vid_bit_arr = np.frombuffer(vid_bit_str, np.uint8)
        '''

        if self.logo_detector:

            dir = os.path.split(self.vid_f_in)[0]
            dir = dir.rsplit(os.sep,1)[0]
            dir = os.path.join(dir,'Ads')
            ad_path = {}
            for fpath in glob.glob(dir+'/*.rgb'):
                name = os.path.split(fpath)[1]
                name = os.path.splitext(name)[0]
                name = name.split('_')[0]
                if not ad_path.get(self.label_to_logo_map[name]):
                    ad_path[self.label_to_logo_map[name]] = fpath

            # print(list(ad_path.keys()))
            sampling_rate, audio_buf = wavfile.read(self.aud_f_in)
            FPS = 30
            aud_f_size = int(sampling_rate / FPS)  # 48,000/30 = 1600
            audio = []

            ads_bit_arrs = []
            new_ad = None
            c = 1
            # print(result)
            for _i in range(len(result)):
                i = result[_i]
                if i[-1] == 1:
                    if self.shot_to_logo.get(i[0]):
                        ad_in_shot = self.shot_to_logo.get(i[0])
                        new_ad_list = list(ad_path.keys())

                        if ad_in_shot.get((new_ad_list[0])) and ad_in_shot.get((new_ad_list[1])):
                            if ad_in_shot[new_ad_list[0]] > ad_in_shot[new_ad_list[0]]:
                                new_ad = new_ad_list[0]

                            else:
                                new_ad = new_ad_list[1]

                        if ad_in_shot.get(new_ad_list[0]) and ad_in_shot.get(new_ad_list[0]):
                            new_ad = new_ad_list[0]
                        if ad_in_shot.get(new_ad_list[1]):
                            new_ad = new_ad_list[1]

                    temp = vid_bit_arr[((i[1] - 1) * self.VID_F_SIZE): (i[2] * self.VID_F_SIZE)]

                    # TODO: detection on frames in temp prior to write
                    # output_video_file.write(bytearray(temp))
                    ads_bit_arrs.append(temp)

                    print("[INFO] finished processing audio sequence {} of {}".format(c, len(result)))
                    c += 1

                    audio += audio_buf[((i[1] - 1) * aud_f_size):(i[2] * aud_f_size)].tolist()

                else:
                    if ad_path.get(new_ad):
                        print(ad_path[new_ad])
                        ad = open(ad_path[new_ad], 'rb')
                        temp = ad.read()
                        temp = np.frombuffer(temp, np.uint8)
                        ads_bit_arrs.append(temp)
                        ad.close()

                        ad_audio = os.path.splitext(ad_path[new_ad])[0]
                        ad_audio += '.wav'
                        s, ad_audio = wavfile.read(ad_audio)
                        audio += ad_audio.tolist()

                        new_ad = None

            # wavfile.write(self.aud_f_out, sampling_rate, np.array(audio).astype(np.int16))
            # output_video_file.close()
            self.sampling_rate = sampling_rate
            self.audio = np.array(audio).astype(np.int16)

            ads_bit_arr = np.concatenate(ads_bit_arrs, axis=0, dtype=np.uint8)
            return ads_bit_arr

        else:
            sampling_rate, audio_buf = wavfile.read(self.aud_f_in)
            FPS = 30
            aud_f_size = int(sampling_rate / FPS)  # 48,000/30 = 1600
            audio = []

            ads_bit_arrs = []
            c = 1
            for _i in range(len(result)):
                i = result[_i]
                if i[-1] == 1:
                    temp = vid_bit_arr[((i[1] - 1) * self.VID_F_SIZE) : (i[2] * self.VID_F_SIZE)]

                    # TODO: detection on frames in temp prior to write
                    #output_video_file.write(bytearray(temp))
                    ads_bit_arrs.append(temp)

                    print("[INFO] finished processing audio sequence {} of {}".format(c, len(result)))
                    c += 1

                    audio += audio_buf[((i[1]-1)*aud_f_size):(i[2]*aud_f_size)].tolist()

            #wavfile.write(self.aud_f_out, sampling_rate, np.array(audio).astype(np.int16))
            #output_video_file.close()
            self.sampling_rate = sampling_rate
            self.audio = np.array(audio).astype(np.int16)

            ads_bit_arr = np.concatenate(ads_bit_arrs, axis=0, dtype=np.uint8)
            return ads_bit_arr

    def label_logos(self, vid_bit_arr, score_thresh=0.7, discovery=False):
        out_bit_arr = np.zeros(vid_bit_arr.shape, dtype=np.uint8)
        found_logo = False
        label_to_confidence = {}

        for i in range(vid_bit_arr.shape[0]//self.VID_F_SIZE):
            img = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_DEPTH), dtype=np.uint8)
            for d in range(self.IMG_DEPTH):
                channel = np.reshape(vid_bit_arr[d * self.IMG_AREA : (d + 1) * self.IMG_AREA],
                                     (self.IMG_HEIGHT, self.IMG_WIDTH))
                img[:, :, (2 - d)] = channel

            # detector res returns an list of length NUM_CLASSES (number of target classes for detector)
            # Each element in this list is a list of candidate detections for that class
            # Each candidate detection is an array of 5 entries.
            # The first 4 entries of this array are the corners of the bboxes
            # the last entry is the confidence score. We use the confidence score to identify whether or not
            # a logo was detected
            detector_res = self.logo_detector.run_inference(img)

            # Log if a logo was found or not
            '''
            if discovery:
                for j in range(len(detector_res)):
                    if found_logo:
                        break
                    if detector_res[j].shape[0] == 0:
                        continue
                    for det in detector_res[j]:
                        if det[-1] >= score_thresh:
                            found_logo = True
                            break
            '''
            #'''
            for j in range(len(detector_res)):
                if detector_res[j].shape[0] == 0:
                    continue
                for k in range(len(detector_res[j])):
                    det = detector_res[j][k]
                    if det[-1] >= score_thresh:
                        found_logo = True
                        label = self.logo_detector.cfg.CLASSES[j]
                        if label not in label_to_confidence:
                            label_to_confidence[label] = det[-1]
                        else:
                            label_to_confidence[label] = max(det[-1], label_to_confidence[label])
                        
                        
                        cv2.rectangle(img, (int(det[0]), int(det[1])),
                                      (int(det[2]), int(det[3])), (0, 255, 0), 1)
                        cv2.putText(img, "{}, {}".format(self.logo_detector.cfg.CLASSES[j], det[-1]),
                                    (int(det[0]) + 10, int(det[3]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0), 1, 2)
            #'''
            #img_out = self.logo_detector.model.show_result(img, detector_res, show=False, score_thr=score_thresh)
            #out_bit_arr[i * self.VID_F_SIZE : (i + 1) * self.VID_F_SIZE] = self.bit_arr_from_frame(img_out)
            out_bit_arr[i * self.VID_F_SIZE : (i + 1) * self.VID_F_SIZE] = self.bit_arr_from_frame(img)
            #out_bit_arr = self.bit_arr_from_frame(img_out)

        # TODO: Also return what was labeled, if anything

        return (out_bit_arr, found_logo, label_to_confidence)

    #def process_logos(self, vid_bit_arr):
    def process_logos(self):
        video = open(self.vid_f_in, 'rb')
        vid_bit_str = video.read()
        vid_bit_arr = np.frombuffer(vid_bit_str, np.uint8).copy()

        if self.logo_detector:
            LOGO_QUERY_INTERVAL = self.FRAME_RATE
            frames_with_logo = []
            frame_to_label_and_confidence = {}
            for i in tqdm(np.arange(0, vid_bit_arr.shape[0]//self.VID_F_SIZE, LOGO_QUERY_INTERVAL),
                          desc="Logo Discovery"):
                #print("[INFO] initial pass of logo detection on frame {}".format(i))
                label_res = self.label_logos(vid_bit_arr[i * self.VID_F_SIZE : (i + 1) * self.VID_F_SIZE], discovery=True)
                vid_bit_arr[i * self.VID_F_SIZE : (i + 1) * self.VID_F_SIZE] = label_res[0]
                found_logo = label_res[1]
                label_to_confidence = label_res[-1]
                #vid_bit_arr[i * self.VID_F_SIZE : (i + 1) * self.VID_F_SIZE], found_logo = \
                #        self.label_logos(vid_bit_arr[i * self.VID_F_SIZE : (i + 1) * self.VID_F_SIZE])
                if found_logo:
                    frames_with_logo.append(i)
                    frame_to_label_and_confidence[i] = label_to_confidence

            self.frame_to_label_and_confidence = frame_to_label_and_confidence
            consecutive_detected_intervals = []
            interval = [frames_with_logo[0], None]
            for i in range(1, len(frames_with_logo)):
                if frames_with_logo[i] - frames_with_logo[i - 1] == LOGO_QUERY_INTERVAL:
                    interval[-1] = frames_with_logo[i]
                    continue
                if interval[-1] is not None:
                    consecutive_detected_intervals.append(tuple(interval))
                interval = [frames_with_logo[i], None]
            if interval[-1] is not None:
                consecutive_detected_intervals.append(tuple(interval))

            # Refinement pass: Compute bboxes in frames within selected intervals
            # print("[INFO] Running bbox inference for intervals {}".format(consecutive_detected_intervals))
            # for i in tqdm(consecutive_detected_intervals, desc="Logo Refinement"):
            #     for j in range(i[0] + 1, i[-1]):
            #         label_res = self.label_logos(vid_bit_arr[j * self.VID_F_SIZE : (j + 1) * self.VID_F_SIZE])
            #         vid_bit_arr[j * self.VID_F_SIZE : (j + 1) * self.VID_F_SIZE] = label_res[0]
            #     #label_res = self.label_logos(vid_bit_arr[(i[0] + 1) * self.VID_F_SIZE : i[-1] * self.VID_F_SIZE])
            #     #vid_bit_arr[(i[0] + 1) * self.VID_F_SIZE : i[-1] * self.VID_F_SIZE] = label_res[0]

        return vid_bit_arr

    def shot_logo_sync(self, shots):
        if self.logo_detector:
            shot_to_logo = {}
            for frame, logo in self.frame_to_label_and_confidence.items():
                shot_number = 0
                while not (shots[shot_number][1] <= frame <= shots[shot_number][2]):
                    shot_number+=1

                shot_number = shots[shot_number][0]
                logo_name, logo_conf = list(logo.items())[0]
                # print(shot_number, logo, shot_to_logo.get(shot_number))
                if shot_to_logo.get(shot_number):
                    if shot_to_logo[shot_number].get(logo_name):
                        if shot_to_logo[shot_number][logo_name] < logo_conf:
                            shot_to_logo[shot_number][logo_name] = logo_conf
                    else:
                        shot_to_logo[shot_number][logo_name] = logo_conf
                    # if shot_to_logo[shot_number][1] > logo[1]:
                    #     shot_to_logo[shot_number] = logo
                else:
                    shot_to_logo[shot_number] = {logo_name:logo_conf}

            self.shot_to_logo = shot_to_logo

    def write_output(self, vid_bit_arr):
        output_video_file = open(self.vid_f_out, "wb")
        output_video_file.write(bytearray(vid_bit_arr))
        output_video_file.close()
        wavfile.write(self.aud_f_out, self.sampling_rate, self.audio)

# Function to read in new ad and output the frames in matrix form, as well as the audio
# directory: the directory where the ad is
# logo: the logo detected to find the correct ad
# def get_new_ad(directory, logo):
#     backslashStr = "/" if directory[-1] != '/' else ""
#     ad_vid_path = f"{directory}{backslashStr}{logo}_Ad_15s_.rgb"
#     ad_wav_path = f"{directory}{backslashStr}{logo}_Ad_15s_.wav"
#
#     # Read Audio
#     sampling_rate, audio_buffer = wavfile.read(ad_wav_path)
#
#     # Read Video
#     vid_f = open(ad_vid_path, 'rb')
#     IMG_HEIGHT = 270
#     IMG_WIDTH = 480
#     IMG_DEPTH = 3
#     IMG_AREA = IMG_WIDTH*IMG_HEIGHT
#     FRAME_SIZE = IMG_HEIGHT*IMG_HEIGHT*IMG_DEPTH
#
#     one_frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), dtype=np.uint8)
#     img_buffer = []
#
#     bit_str = vid_f.read(FRAME_SIZE)
#     while bit_str and len(bit_str) < FRAME_SIZE:
#             bit_arr = np.frombuffer(bit_str, np.uint8)
#             for d in range(IMG_DEPTH):
#                 channel = np.reshape(bit_arr[d * IMG_AREA:(d+1) * IMG_AREA],
#                                     (IMG_HEIGHT, IMG_WIDTH))
#                 one_frame[:,:,d] = channel
#             img_buffer.append(one_frame)
#
#     print("Debug: Finished reading advertisement")
#     vid_f.close()
#
#     return audio_buffer, img_buffer
