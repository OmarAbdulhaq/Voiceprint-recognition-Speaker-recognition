import os
from flask import Flask
from flask import request, jsonify
# from tools.sql_code import SQLCODE
# from tools.db_tool import MySQLPro
from voice_recognize.extract_feature.feature_compute import return_results
from voice_recognize.extract_feature.feature_compute import get_evaluate_enrollment_feature
app = Flask(__name__)

register_data_path = r'F:\voiceprint\data\register'
Enrollment_png_path = r'F:\voiceprint\data\Enrollment_png'
Enrollment_Feature_path = r'F:\voiceprint\data\Enrollment_Feature'
Evaluation_saved_png_path = r'F:\voiceprint\data\Evaluate_Feature'
# mysqlpro = MySQLPro()  # 操作数据库


# 一对多的验证
# @app.route('/feature_check')
def feature_check(voice_path):  # 传进来的是用户的语音路径，与声纹库里面的语音进行对比，得出结果。
    """
    voice_path: 存放的语音路径
    :param voice_path:
    :return:
    """
    # voice_path的命名方式，手机号_时间戳。手机号可以唯一的确定出一个人？ 换了手机号怎么办？
    assert isinstance(voice_path, str) and hasattr(voice_path, '__len__') and len(voice_path) > 0
    if '\\' in voice_path:
        voice_path = voice_path.replace('\\', '/')
    the_label = voice_path.split('/')[-1].split('.')[0]  # 获取音频文件名
    #  get_evaluate_enrollment_feature(the_label_time, the_label, saved_png_path, saved_feature_path, _wav_file_path)
    evaluation_feature = get_evaluate_enrollment_feature(the_label_time=None,
                                                         the_label=the_label,
                                                         saved_png_path=Evaluation_saved_png_path,
                                                         saved_feature_path=None,  # 不保存用户语音特征
                                                         _wav_file_path=voice_path,
                                                         ab='SWTZ_SWYC')
    if evaluation_feature is not None:
        label = return_results(Enrollment_Feature_path, evaluation_feature)
        print('label:', label)
        # return result
        return label


# @app.route('/register')
def register(video_path, abnormal='SWTZ_SWYC'):  # 注册  时间---切出音频
    """
    :param video_path: 录音路径
    :param abnormal: 声纹标志。
    :return:
    """
    # 1、先提取语音声纹并且保存到服务器,注册之前先要判断这个人的声纹是否已经被注册过。
    try:
        if '\\' in video_path:
            video_path = video_path.replace('\\', '/')
        the_label = video_path.split('/')[-1].split('.')[0]  # 取录音路径中录音文件名
        # 注册不需要返回提取到的声纹。
        png_feature_path = get_evaluate_enrollment_feature(the_label_time=None,
                                                           the_label=the_label,
                                                           saved_png_path=Enrollment_png_path,
                                                           saved_feature_path=Enrollment_Feature_path,  # Saved
                                                           _wav_file_path=video_path,
                                                           ab=abnormal)
        if (png_feature_path is None) or (isinstance(png_feature_path, list)
                                          and len(png_feature_path) != 2):  # 声纹提取失败，或没有返回值
            return 'false'
        if isinstance(png_feature_path, str):
            print(png_feature_path)
            return png_feature_path
        # 2、将结果写入到声纹表里面
        # png_path, feature_path = png_feature_path[0], png_feature_path[1]  # 图片和声纹路径
        # insert_sql = SQLCODE.insert_into_echo_risk_voice_log([the_label, video_path, png_path, feature_path],
        #                                                      table_name='ECHO_RISK_VOICE_LOG')
        # mysqlpro.insert_mysql(insert_sql, limit=None)  # 插入到数据库
        return 'true'
    except InterruptedError:
        return 'false'


if __name__ == '__main__':
    # so = FeatureCompute()
    # register(os.path.join(register_data_path, '1529374505292.wav'))
    # 验证的时候用户的声纹图片需要存起来么。
    feature_check(os.path.join(register_data_path, '13301168738.wav'))
    # app.run(host='192.168.99.214', port=18001)
    # app.run(host='127.0.0.1', port=18001)










