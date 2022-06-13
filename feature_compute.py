import os
import cv2
import numpy as np
import mxnet as mx
from PIL import Image
from pydub import AudioSegment
from collections import namedtuple
FRAMERATE = 16000  # 16k语音
HAMMING_TIME = 0.025  # 汉明窗  25ms
STEP_TIME = 0.010  # 帧移   10ms
SPEC_THRESH = 4  # 阈值
IMAGE_WIDTH = 512  # width 宽度


class InferParameter(object):
    def __init__(self):
        self.batch_size = 1
        self.shape = (3, 512, 300)
        self.load_path = r'../../train_resnet_model/resnet-18-0'
        self.path_checkpoint = 50
        self.data_shapes = (self.batch_size, 3, 512, 300)
        self.extract_feature_layer_name = 'relu_fc1_output'


# png---->npy
def person_feature(one_png):
    # 图片的存储路径如：\root\voiceprint\data\Evaluate_Feature\11223344556\11223344556.png
    print('one_png:', one_png)
    img = cv2.imread(one_png)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 512))/255
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    infer_param = InferParameter()
    sym, arg_params, aux_params = mx.model.load_checkpoint(infer_param.load_path, infer_param.path_checkpoint)
    all_layers = sym.get_internals()
    fe_sym = all_layers[infer_param.extract_feature_layer_name]
    fe_mod = mx.mod.Module(symbol=fe_sym, context=[mx.cpu()], label_names=None)
    fe_mod.bind(for_training=False, data_shapes=[('data', infer_param.data_shapes)])
    fe_mod.set_params(arg_params, aux_params)
    Batch = namedtuple('Batch', ['data'])
    try:
        fe_mod.forward(Batch([mx.nd.array(img)]))
        output = fe_mod.get_outputs()[0].asnumpy()
        print('output:', output)
        return output
    except IOError:
        return None


def save_feature(save_feature_path, ab, label, the_feature):
    # save_feature(saved_feature_path, ab, the_label, p_features)
    try:
        new_path = os.path.join(save_feature_path, ab)
        if not os.path.exists(new_path):
            os.mkdir(new_path)  # 创建新的文件

        if not os.path.exists(new_path + "/" + label):
            os.mkdir(new_path + "/" + label)
        feature_path = os.path.join(new_path, label, label)
        np.save(feature_path, the_feature)  # 保存声纹特征
        return '.'.join([feature_path, 'npy'])
    except IOError:
        return None


# 通过 cosin计算相似度
# 计算得分
def get_enrollment_persons(path):
    assert os.path.exists(path) is True, '文件不存在'
    count = 0
    ss = os.listdir(path)
    for gg in ss:
        ff = os.listdir(path + "/" + gg)
        for _ in ff:
            count += 1
    return count


def get_all_enrollment_person_feature(path):
    assert os.path.exists(path) is True, '文件不存在'
    model = []
    ss = os.listdir(path)
    for gg in ss:
        ff = os.listdir(path + "/" + gg)
        for feature in ff:
            model.append(np.load(path + "/" + gg + "/" + feature))
    return model


def overlap(x, window_size, window_step):
    assert window_size % 2 == 0, "Window size must be even!"
    append = np.zeros((window_size - len(x) % window_size))
    X = np.hstack((x, append))
    ws = int(window_size)
    ss = int(window_step)
    valid = len(X) - ws
    nw = valid // ss
    out = np.ndarray((nw, 1024), dtype=X.dtype)
    for i in range(nw):
        start = i * ss
        stop = start + ws
        tmp = X[start: stop]
        tmp = np.hamming(ws) * tmp
        sig1024 = np.hstack((tmp, np.zeros(1024 - len(tmp))))
        out[i] = np.fft.fft(sig1024)
    return out[:, :512]


def stft(x, fftsize=128, step=65, mean_normalize=True):
    if mean_normalize:
        x -= x.mean()
    x = overlap(x, fftsize, step)
    return x


def pretty_spectrogram(audio_data, log=True, thresh=5,
                       fft_size=512, step_size=64):
    specgram = np.abs(stft(audio_data, fftsize=fft_size, step=step_size))
    if log:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
        specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
        specgram += thresh
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold
    return specgram


def read_voice(audio_file, ext='.mp3'):
    if ext.upper() == '.mp3':
        audio = AudioSegment.from_file(audio_file, 'mp3')
    elif ext.upper() == '.flac':
        audio = AudioSegment.from_file(audio_file)
    elif ext.upper() == '.WAV':
        audio = AudioSegment.from_file(audio_file)[1300:4800]
    else:
        raise Exception
    audio = audio.set_frame_rate(FRAMERATE)  # 读进来音频文件全部转成16K
    if audio.sample_width == 2:
        data = np.fromstring(audio._data, np.int16)  # 位数表示
    elif audio.sample_width == 4:
        data = np.fromstring(audio._data, np.int32)
    else:
        raise Exception
    x = []
    for chn in range(audio.channels):
        x.append(data[chn::audio.channels])  # [::]相当于后面的是步长step
    x = np.array(x).T  # 转置运算
    return FRAMERATE, x


def voice2image(param):
    # voice2image((wav_file_path, color, to_save_file, ab))
    print('param:', param)
    fft_size = int(FRAMERATE * HAMMING_TIME)  # 400
    step_size = int(FRAMERATE * STEP_TIME)  # 160
    audio, color, out_dir, abnormal = param  # 之前是一个元组，打包进来的
    if '\\' in audio:
        audio = audio.replace('\\', '/')
    slice_list = audio.split('/')  # [Evaluate_Test,SA1.WAV]
    assert len(slice_list) >= 2, 'data must have folder'
    frame_rate, wave_data = read_voice(audio, ext=os.path.splitext(audio)[1])
    if len(wave_data.shape) > 1:
        wave_data = np.mean(wave_data, axis=1)
    wav_spectrogram = pretty_spectrogram(wave_data.astype('float64'),
                                         fft_size=fft_size,
                                         step_size=step_size,
                                         log=True,
                                         thresh=SPEC_THRESH)
    wav_spectrogram = wav_spectrogram / np.max(wav_spectrogram) * 255.0
    spect_width = wav_spectrogram.shape[0]
    batch = 1
    if spect_width > 512:
        batch = int(wav_spectrogram.shape[0] / IMAGE_WIDTH)
    if spect_width < 300:
        return None  # 时间不够
    try:
        if abnormal is not None:
            path = os.path.join(out_dir, abnormal, slice_list[-1].split('.')[0])  # 图片根目录/类别/图片.png
        else:
            path = os.path.join(out_dir, slice_list[-1].split('.')[0])  # 图片根目录/类别/图片.png
        for i in range(0, batch * IMAGE_WIDTH, IMAGE_WIDTH):
            slice_spectrogram = wav_spectrogram[i: i + IMAGE_WIDTH, :]
            spectrogram = slice_spectrogram.astype(np.uint8).T
            spectrogram = np.flip(spectrogram, 0)  # 翻转操作,倒排
            if not os.path.exists(path):
                os.makedirs(path)
            if color:
                new_im = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
                cv2.imwrite('%s/%s.png' % (path, slice_list[-1].split('.')[0]), new_im)

            else:
                new_im = Image.fromarray(spectrogram)
                new_im.save('%s/%s.png' % (path, i))
        return '.'.join([path, slice_list[-1].split('.')[0], 'png'])  # 返回图片路径
    except InterruptedError:
        return None


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('AUDIO_LIST', type=str, help='voice lst file path')
    parser.add_argument('OUT_DIR', type=str, help='voice output path')
    parser.add_argument('--color', type=bool, default=False, help='the output picture is gay or color')
    return parser.parse_args()


def get_wav(wav_file_path, to_save_file, ab, color=True):
    """
    语音转图片
    :param wav_file_path:  原始语音路径
    :param to_save_file:  图片存放位置
    :param ab:  声纹类别
    :param color:  图片是否使用伪彩色
    :return:
    """
    print('问题:', (wav_file_path, to_save_file, ab))
    return voice2image((wav_file_path, color, to_save_file, ab))


def get_cosin_score(model, test_feature):
    numclass = len(model)  # 注册的说话人的个数。
    score_vectore = np.zeros((numclass, 1))
    for index in range(numclass):  # 其实就是每个人的录音就128个   对于每一段语音。，都要对这段语音可能属于的类别进行判断
        enrollmented_model = model[index]  # 得到当前类别标签对应的说话人的特征。
        n_feature = np.sqrt(test_feature)
        m_feature = np.sqrt(enrollmented_model)
        # # calculate norm
        m_norm = m_feature / np.sqrt(np.sum(np.square(m_feature), axis=1, keepdims=True))
        n_norm = n_feature / np.sqrt(np.sum(np.square(n_feature), axis=1, keepdims=True))
        similar_score = np.dot(n_norm, m_norm.T)
        score_vectore[index] = similar_score
    return score_vectore


# WAV----->png——————>npy 1024
def get_evaluate_enrollment_feature(the_label_time, the_label, saved_png_path, saved_feature_path, _wav_file_path, ab=None):
    """

    :param the_label_time:
    :param the_label:
    :param saved_png_path:
    :param saved_feature_path:
    :param _wav_file_path:
    :param ab: 是声纹的类别标识，诈骗，骚扰，传销，营销等是人为标注的结果
    :return:
    """
    try:
        try:
            _png_path = get_wav(_wav_file_path, saved_png_path, ab)  # wav->png
            if _png_path is None:  # 图片提取错误
                return None
            if the_label_time is not None:
                p_features = person_feature(os.path.join(saved_png_path, ab, the_label_time, the_label_time + ".png"))
            else:
                print('data path:', (saved_png_path, ab, the_label, the_label + ".png"))
                p_features = person_feature(os.path.join(saved_png_path, ab, the_label, the_label + ".png"))  # 1024
                print(p_features)
            if p_features is None:  # 声纹特征提取不成功。
                return None
            # 不要马上存，得到特征之后，与已经存在的声纹进行对比
            if saved_feature_path is not None:  # 特征文件下面也会有4个类别的子文件夹
                exist = return_results(saved_feature_path, p_features)  # 所有特征的存放路径， 当前的用户声纹。
                print('检测是否被注册过', exist)
                if exist.__eq__('normal'):  # 之前没有被注册
                    feature_path = save_feature(saved_feature_path, ab, the_label, p_features)  # 保存声纹特征
                    if feature_path is None:
                        return None
                    if '\\' in _png_path:
                        _png_path = _png_path.replace('\\', '/')
                    if '\\' in feature_path:
                        feature_path = feature_path.replace('\\', '/')
                    return [_png_path, feature_path]  # 返回语音图片路径，声纹路径
                else:
                    return 'had been enrollmented'  # 此人的声纹已经注册
            # else:
            #     return None  # 没有得到声纹
            return p_features
        except InterruptedError:
            return None
    except InterruptedError:
        return None


# 返回1对多的结果
def return_results(saved_feature_path, evaluated_feature):
    """
    1：N 判别用户声纹
    :param saved_feature_path:  用户声纹存储的根目录
    :param evaluated_feature:   需要比对的用户声纹
    :return: abnormal
    """
    cat_labels = os.listdir(saved_feature_path)  # 声纹所属的大类
    # enrollment_features = list(map(lambda cur_enrollment_fea: np.load(cur_enrollment_fea),
    #                                list(map(lambda cur_fea_path: os.path.join(saved_feature_path, cur_fea_path),
    #                                         cat_labels))))
    # 将一个很大的多维列表转换成数组，效率可能有问题，这个地方今后可以斟酌
    # enrollment_features = np.array(list(map(lambda cur_enrollment_fea_path: np.load(cur_enrollment_fea_path),
    #                                list(map(lambda cur_fea_name: os.listdir(os.path.join(path, cur_fea_name)),
    #                                         cat_labels))))).ravel()
    ab_normal = 'normal'
    if len(cat_labels) < 1:
        return ab_normal  # 声纹库空
    for cur_cat in cat_labels:
        enrollment_features = list(map(lambda cur_enrollment_fea_path: np.load(os.path.join(saved_feature_path, cur_cat,
                                                                                            cur_enrollment_fea_path,
                                                                                            cur_enrollment_fea_path + '.npy')),
                                       os.listdir(os.path.join(saved_feature_path, cur_cat))))
        if len(enrollment_features) < 1:
            continue  # 基本不会出现
        score_vector = get_cosin_score(enrollment_features[:], evaluated_feature)
        max_score = score_vector.max()  # 最大的分数
        if max_score < 0.78:
            continue
            # return False
        else:
            # print('index:', index)
            # return fgfg[index_count], 'true'
            ab_normal = cur_cat
        if not ab_normal.__eq__('normal'):
            return ab_normal
    return ab_normal


# # 返回1对多的结果
# def return_results(saved_feature_path, evaluated_feature):
#     enrollment_list = []
#     fgfg = os.listdir(saved_feature_path)
#     for gnn in fgfg:
#         uuu = os.listdir(saved_feature_path + "/" + gnn)
#         for yyy in uuu:
#             enrollment_list.append([saved_feature_path + "/" + gnn + "/" + yyy])
#     the_enrollment_features = []
#     for have_Enrollmented_Person in enrollment_list:  # npy的路径
#         the_enrollment_features.append(np.load(have_Enrollmented_Person[0]))
#
#     if len(the_enrollment_features) < 1:
#         return False
#     score_vecter = get_cosin_score(the_enrollment_features[:], evaluated_feature)
#     max_score = 0
#     index_count = 0
#     for index, cur_score in enumerate(score_vecter):
#         if max_score < cur_score[0]:
#             max_score = cur_score[0]
#             index_count = index
#     print('score:', max_score)
#     if max_score < 0.78:
#         # return fgfg[index_count], 'false'
#         return 'false'
#         # return False
#     else:
#         # print('index:', index)
#         # return fgfg[index_count], 'true'
#         return 'true'

# path = '11111111111'
# cat_labels = os.listdir(path)  # 声纹所属的大类
# enrollment_features = np.array(list(map(lambda cur_enrollment_fea: cur_enrollment_fea,
#                                list(map(lambda cur_fea_path: os.listdir(os.path.join(path, cur_fea_path)), cat_labels))))).ravel()
#
# print(enrollment_features)

# gege = np.array([10, 2, 30, 14, 55, 6])
# max_s = gege.max()
# print('max_s:', max_s)