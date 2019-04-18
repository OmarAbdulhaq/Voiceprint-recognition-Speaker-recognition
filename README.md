# Voiceprint-recognition-Speaker-recognition

      It is a complete project of  voiceprint recognition or speaker recognition.The trained models my not be uploaded except the best one.  
So if you are lazy enougth, you can dirrectly run my model, maybe, you should only exchange the model path to satisfy your system.
Noting: the program was wirtten by hand, so may existing some chinses notes in my code. If your system is Contos7.x, u should delete all chineses notes, because Centos may not support chinese. 

To run my model, some dependency should installed ：
 
 MXNET               1.2.0
 PyMySQL             0.9.0
 opencv-python       3.4.1.15
 Pillow              5.2.0
 pydub               0.22.1
 
 example: 
         #  input:13301234567.wav
         feature_check(os.path.join(register_data_path, '13301234567.wav'))
         output:  label: normal
