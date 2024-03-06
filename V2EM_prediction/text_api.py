# import wave, math, contextlib
# import speech_recognition as sr
# from moviepy.editor import AudioFileClip
#
# zoom_video_file_name = "..\\video\\Sample_1\\Sample_1_video.mp4"
# transcribed_audio_file_name = "..\\video\\Sample_1_audio_test.wav"
#
# audioclip = AudioFileClip(zoom_video_file_name)
# audioclip.write_audiofile(transcribed_audio_file_name)
#
# with contextlib.closing(wave.open(transcribed_audio_file_name,'r')) as f:
#     frames = f.getnframes()
#     rate = f.getframerate()
#     duration = frames / float(rate)
# total_duration = math.ceil(duration / 60)
# r = sr.Recognizer()
# for i in range(0, total_duration):
#     with sr.AudioFile(transcribed_audio_file_name) as source:
#         audio = r.record(source, offset=i*60, duration=60)
#     f = open("..\\video\\Sample_1_text_test.txt", "a")
#     f.write(r.recognize_google(audio))
#     f.write(" ")
# f.close()


import openai

# openai.api_base = 'https://35.mctools.online/v1'
openai.api_key = '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'

# file = "..\\video\\Sample_1\\Sample_1_audio.wav"
transcription = openai.Audio.transcribe("whisper-1", open("..\\video\\Sample_1\\Sample_1_audio.wav", "rb"))
# translation = openai.Audio.translate("whisper-1", open("..\\video\\Sample_1\\Sample_1_audio.wav", "rb"))

print(transcription.text)
