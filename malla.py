import os

import cv2
import mediapipe as mp
import tkinter as tk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk

# 1. Configuración de Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Se usa el modelo local que viene junto al script: no se descarga nada.
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_landmarker.task")


# Opciones del detector para video en tiempo real
if not os.path.exists(MODELO_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo local en {MODELO_PATH}. "
        "Coloca face_landmarker.task junto a malla.py."
    )

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELO_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

# Configuración personalizada de los puntos
RADIO_PUNTO = 2              # Aumenta este número para puntos más grandes
COLOR_PUNTO = (255, 120, 0)  # Color en formato BGR (Azul, Verde, Rojo)

ANCHO_VENTANA = 640
ALTO_VENTANA = 480

# Estilos del botón (equivalentes al stylesheet anterior)
ESTILO_INICIAR = {"bg": "#4CAF50", "activebackground": "#45a049"}
ESTILO_DETENER = {"bg": "#F44336", "activebackground": "#da190b"}


class LandmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe Face Mesh - Tkinter")

        self.cap = None
        self.landmarks_activos = False
        self._photo = None  # Referencia para que Tkinter no borre la imagen
        self.landmarker = FaceLandmarker.create_from_options(options)

        # --- Widgets ---

        # Área donde se muestra el video.
        # Se le da tamaño fijo y fondo gris oscuro con mensaje, asi si la
        # cámara falla se ve claramente que no hay señal (no "todo oscuro").
        self.video_label = tk.Label(
            root,
            text="Esperando cámara...",
            width=ANCHO_VENTANA,
            height=ALTO_VENTANA,
            bg="#222222",
            fg="#aaaaaa",
            font=("Arial", 16),
        )
        self.video_label.pack(padx=10, pady=10)

        # Botón de iniciar / detener landmarks
        self.boton = tk.Button(
            root,
            text="Iniciar Landmark",
            command=self.toggle,
            fg="white",
            font=("Arial", 14, "bold"),
            padx=10,
            pady=10,
            **ESTILO_INICIAR,
        )
        self.boton.pack(fill="x", padx=10, pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        # La cámara arranca sola, sin landmarks
        self.iniciar_camara()

    def toggle(self):
        self.landmarks_activos = not self.landmarks_activos
        if self.landmarks_activos:
            self.boton.config(text="Detener Landmark", **ESTILO_DETENER)
        else:
            self.boton.config(text="Iniciar Landmark", **ESTILO_INICIAR)

    def iniciar_camara(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("No se pudo abrir la cámara")
            self.cap = None
            self.video_label.config(text="No se pudo abrir la cámara")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_VENTANA)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_VENTANA)
        self.actualizar_frame()

    def detener_camara(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="", text="Cámara detenida")

    def actualizar_frame(self):
        if self.cap is None:
            return

        success, frame = self.cap.read()
        if not success:
            self.detener_camara()
            return

        # Solo detectar y dibujar la malla cuando el botón la activa
        if self.landmarks_activos:
            # Convertir a RGB (MediaPipe lo requiere)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 3. Detectar la malla (Face Mesh)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # 4. Dibujar la malla en el frame
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    for landmark in face_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])

                        # Dibujar el punto con el nuevo tamaño y color
                        cv2.circle(frame, (x, y), RADIO_PUNTO, COLOR_PUNTO, -1)

        # Mostrar el frame en la ventana de Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(rgb_frame)
        self._photo = ImageTk.PhotoImage(image=imagen)
        self.video_label.config(image=self._photo, text="")

        # Timer que actualiza el frame (~30 fps), equivalente al QTimer
        self.root.after(30, self.actualizar_frame)

    def cerrar(self):
        self.landmarks_activos = False
        self.detener_camara()
        self.landmarker.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LandmarkApp(root)
    root.mainloop()
