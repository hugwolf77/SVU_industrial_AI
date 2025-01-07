import os
# from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress

# https://pytubefix.readthedocs.io/en/latest/user/streams.html

def download_video(video_url):
    yt = YouTube(video_url, on_progress_callback = on_progress)
    print(yt.title)
    # ys = yt.streams.get_highest_resolution()
    ys = yt.streams.filter(only_audio=True).first()
    ys.download('./store')


if __name__ == '__main__':
    video_url = 'https://www.youtube.com/watch?v=U7c4y4qvBGk'
    download_video(video_url)
