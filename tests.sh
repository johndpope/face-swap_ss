python3 inference.py ../reference_videos/0.jpg ../reference_videos/test_video_01.mp4
# no audio
python3 inference.py ../reference_videos/test_frame_01.jpg ../reference_videos/test_video_02-no_audio.mp4
# choose face 1 from 2 on the image
python3 inference.py ../reference_videos/test_frame_02.jpg ../reference_videos/test_video_03-2_faces.mp4
# check .mov
python3 inference.py ../reference_videos/0.jpg ../reference_videos/test_video_04-mov.mov
python3 inference.py ../reference_videos/test_frame_04.jpg ../reference_videos/test_video_05.MOV
# no face in image
python3 inference.py ../reference_videos/test_frame_04.jpg ../reference_videos/sveta_03.mp4
# no face in video
python3 inference.py ../reference_videos/0.jpg ../reference_videos/test_video_06-no_faces.MP4
