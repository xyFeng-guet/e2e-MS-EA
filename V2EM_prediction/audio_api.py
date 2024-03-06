import moviepy.editor as mp

my_clip = mp.VideoFileClip("..\\video\\Sample_1\\Sample_1_video.mp4")
my_clip.audio.write_audiofile("..\\video\\Sample_1_audio_test.wav")

