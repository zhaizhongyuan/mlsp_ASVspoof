import librosa as lr
import numpy as np



def vad(y_t,
        sr,
        power_th=1e-5,
        power_tl=1e-6,
        zcr_t=1e-2,
        nframe_th=3,
        nframe_tl=3,
        frame_length=1024,
        hop_length=256):  # 't' means threshold here
    # calculate zero-crossing rate and power of each frame
    zcrs = lr.feature.zero_crossing_rate(y_t,
                              frame_length=frame_length,
                              hop_length=hop_length,
                              center=False)
    powers = []
    n_frames = np.shape(zcrs)[1]
    n_samples = np.shape(y_t)[0]
    frame_labels = lr.samples_to_frames(list(range(n_samples)),
                                     hop_length=hop_length)
    sample_cursor = 0

    for i in range(n_frames):
        this_frame = y_t[np.where(frame_labels == i)[0]]
        power_this = np.mean(this_frame * this_frame)
        powers.append(power_this)

    # get status of each frame
    status = []
    for i in range(n_frames):
        status.append(0)
        if powers[i] > power_tl or zcrs[0][i] > zcr_t:
            status[i] = 1
            if powers[i] > power_th:
                status[i] = 2

    is_voice = False
    flags = []
    for i in range(n_frames):
        if is_voice:
            flags.append(True)
            if status[i] == 0 and i >= nframe_tl:
                for j in range(nframe_tl):
                    if status[i - j] != 0:
                        break
                    is_voice = False
        else:
            flags.append(False)
            if status[i] == 2 and i >= nframe_th:
                for j in range(nframe_th):
                    if status[i - j] != 2:
                        break
                    is_voice = True

    for i in range(n_frames):
        if flags[i]:
            l_cursor = int(np.where(frame_labels == i)[0][0])
            break
    for i in range(n_frames):
        if flags[n_frames - i - 1]:
            r_cursor = int(np.where(frame_labels == n_frames - i - 1)[0][-1])
            break
    y_t = y_t[l_cursor:r_cursor]

    return y_t