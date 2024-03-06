import cv2

video = cv2.VideoCapture(0)     # 调用摄像头，
judge = video.isOpened()    # 查看摄像头是否打开
frame_size = (int(video.get(3)), int(video.get(4)))     # 获取分辨率

FPS = video.get(5)  # 摄像头帧率
print("FPS: ", FPS)

code = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 30
filename = 'Sample_x_video.mp4'
out = cv2.VideoWriter(filename, code, fps, frame_size, isColor=True)

if not(out.isOpened()):
    print("out is not opened")

while judge:
    ret, frame = video.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
