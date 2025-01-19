import yt_dlp
import re

def download_video_yt(url, path):
    ydl_opts = {
        'outtmpl': f'{path}/%(title)s.%(ext)s',  # Define la plantilla de salida
        'format': 'best',  # Descarga el mejor formato disponible
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])  # Descarga el video
        print("Download complete!")  # Mensaje de éxito
    except Exception as e:
        print("ERROR:", e)  # Captura y muestra cualquier error que ocurra

if __name__ == "__main__":
    url_youtube = input("Dame el url YT: ")  # Solicita al usuario la URL del video
    if re.match(r'(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$', url_youtube):  # Valida la URL
        path = input("Give the path (default actual path): ")  # Solicita el camino para guardar el video
        if not path:  # Si no se proporciona una ruta, usa la ruta por defecto
            path = r'C:\Users\jacob\Downloads\Detect_person\download'
        download_video_yt(url_youtube, path)  # Llama a la función para descargar el video
    else:
        print("Por favor, introduce una URL válida de YouTube.")  # Mensaje de error si la URL no es válida
