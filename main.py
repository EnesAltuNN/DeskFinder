import cv2
import numpy as np
from ultralytics import YOLO
import json
import os


# AYARLAR VE SABİTLER
# SETTINGS AND CONSTANTS


model = YOLO("yolo11s.pt")

# Tespit edilecek hedef sınıflar
# Target classes to be detected
TARGET_CLASSES = [
    0,  # insan-person
    24, # sırt çantası-backpack
    26, # el çantası-handbag
    39, # şişe-bottle
    41, # fincan-cup
    63, # dizüstü bilgisayar-  laptop
    73  # kitap-book
]

# Güven Eşiği (Confidence Threshold)
# Modelin bir nesneyi "tespit ettim" demesi için gereken minimum güven skoru.
# Minimum confidence score required for the model to register a detection.
CONF_THRESHOLD = 0.40

# Yapılandırma dosyasının adı (Masa koordinatları burada saklanacak)
# Configuration file name (Desk coordinates will be stored here)
CONFIG_FILE = "config.json"

# Varsayılan (default) koordinatlar
# Sadece config.json bulunamazsa bu kullanılacak
# Default coordinates, used only if config.json is not found
DEFAULT_MASA_ROIS_COORDS = {
    "Masa_1/Desk_1": [(10, 500), (5, 1000), (305, 400), (500, 600)]
}

# Bu değişkenler 'load_config' tarafından doldurulacak
# These variables will be populated by 'load_config'
MASA_ROIS_COORDS = {}
MASA_ROIS_NP = {}

# Her masa için gereken köşe noktası sayısı
# Number of corner points required for each table
POINTS_PER_TABLE = 4

# Programın anlık durumunu (state) tutacak sözlük
# Dictionary to hold the application's current state
APP_STATE = {
    "is_in_edit_mode": False,
    "editing_table_name": None,
    "temp_clicks": []
}


# YARDIMCI FONKSİYONLAR
# HELPER FUNCTIONS

def save_config(data):
    """
    Verilen sözlüğü (data) config.json dosyasına yazar.
    Saves the given dictionary (data) to the config.json file.
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Bilgi: Yapılandırma '{CONFIG_FILE}' dosyasına kaydedildi.")
        print(f"Info: Configuration saved to '{CONFIG_FILE}'.")
    except Exception as e:
        print(f"HATA: Yapılandırma kaydedilemedi! Hata: {e}")
        print(f"ERROR: Could not save configuration! Error: {e}")

def load_config():
    """
    config.json dosyasını yükler. Eğer yoksa, varsayılanı oluşturur.
    Loads config.json. If it doesn't exist, creates it using defaults.
    """
    global MASA_ROIS_COORDS, MASA_ROIS_NP

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                MASA_ROIS_COORDS = json.load(f)
            print(f"Bilgi: Yapılandırma '{CONFIG_FILE}' dosyasından yüklendi.")
            print(f"Info: Configuration loaded from '{CONFIG_FILE}'.")
        except Exception as e:
            print(f"HATA: '{CONFIG_FILE}' okunurken hata oluştu. Varsayılanlar kullanılıyor. Hata: {e}")
            print(f"ERROR: Failed to read '{CONFIG_FILE}'. Using defaults. Error: {e}")
            MASA_ROIS_COORDS = DEFAULT_MASA_ROIS_COORDS
            save_config(MASA_ROIS_COORDS)
    else:
        print(f"Bilgi: '{CONFIG_FILE}' bulunamadı. Varsayılan koordinatlar kullanılıyor ve yeni dosya oluşturuluyor.")
        print(f"Info: '{CONFIG_FILE}' not found. Creating new file with default coordinates.")
        MASA_ROIS_COORDS = DEFAULT_MASA_ROIS_COORDS.copy()
        save_config(MASA_ROIS_COORDS)

    MASA_ROIS_NP.clear()
    for name, coords in MASA_ROIS_COORDS.items():
        MASA_ROIS_NP[name] = np.array(coords, dtype=np.int32)

def find_coords(event, x, y, flags, params):
    """
    Mouse sol tuşuna tıklandığında koordinatları işler.
    Handles coordinates when the left mouse button is clicked.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if not params["is_in_edit_mode"]:
            print(f"TIKLANAN KOORDINAT (Mod: Normal): ({x}, {y})")
            print(f"CLICKED COORDINATE (Mode: Normal): ({x}, {y})")
        else:
            # Düzenleme modundaysa, tıklamayı geçici listeye ekle
            # If in edit mode, add the click to the temporary list
            params["temp_clicks"].append((x, y))
            print(f"Düzenleme modu: Nokta {len(params['temp_clicks'])}/{POINTS_PER_TABLE} eklendi: ({x}, {y})")
            print(f"Edit mode: Point {len(params['temp_clicks'])}/{POINTS_PER_TABLE} added: ({x}, {y})")

            # Yeterli noktaya ulaşıldıysa, kaydet ve modu kapat
            # If enough points are clicked, save and exit mode
            if len(params["temp_clicks"]) == POINTS_PER_TABLE:
                table_name = params["editing_table_name"]
                print(f"*** {table_name} İÇİN {POINTS_PER_TABLE} NOKTA TAMAMLANDI. KOORDİNATLAR GÜNCELLENDİ. ***")
                print(f"*** {POINTS_PER_TABLE} POINTS COMPLETED FOR {table_name}. COORDINATES UPDATED. ***")

                MASA_ROIS_COORDS[table_name] = params["temp_clicks"]
                MASA_ROIS_NP[table_name] = np.array(params["temp_clicks"], dtype=np.int32)

                save_config(MASA_ROIS_COORDS)

                params["is_in_edit_mode"] = False
                params["editing_table_name"] = None
                params["temp_clicks"] = []

def check_occupancy(results, rois):
    """
    YOLO sonuçlarını ve ROI'leri analiz eder.
    Analyzes YOLO results and ROIs.
    """
    masa_durumlari = {name: "BOS / AVAILABLE" for name in rois.keys()}
    inside_indices = []

    try:
        boxes = results[0].boxes.cpu().numpy()
    except:
        return masa_durumlari, inside_indices

    if boxes is None or len(boxes) == 0:
        return masa_durumlari, inside_indices

    for i, box in enumerate(boxes):
        center_x = int(box.xywh[0][0])
        center_y = int(box.xywh[0][1])

        is_inside_any_roi = False

        for name, poly in rois.items():
            if cv2.pointPolygonTest(poly, (center_x, center_y), False) >= 0:
                masa_durumlari[name] = "DOLU / OCCUPIED"
                is_inside_any_roi = True
                break

        if is_inside_any_roi:
            inside_indices.append(i)

    return masa_durumlari, inside_indices


# ANA PROGRAM (MAIN)
# MAIN PROGRAM

def main():

    # Yapılandırmayı yükle
    # Load configuration
    load_config()

    # Kamera kaynağını ayarla (0 = dahili, 1+ = harici)
    # Set camera source (0 = built-in, 1+ = external)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Hata: Kamera açılamadı. 'cap = cv2.VideoCapture(0)' satırını kontrol edin.")
        print("Error: Could not open camera. Check 'cap = cv2.VideoCapture(0)' line.")
        return

    window_name = "DeskFinder - Kutuphane Doluluk Paneli / Library Occupancy Panel"
    cv2.namedWindow(window_name)

    # Mouse tıklama dinleyicisini bağla
    # Set the mouse callback listener
    cv2.setMouseCallback(window_name, find_coords, APP_STATE)

    print("Kamera açıldı. Çıkmak için 'q' tuşuna basın.")
    print("Camera opened. Press 'q' to quit.")
    print(f"Bilgi: Sadece %{CONF_THRESHOLD * 100:.0f}'in üzerindeki nesneler gösterilecek.")
    print(f"Info: Only objects with > %{CONF_THRESHOLD * 100:.0f} confidence will be shown.")
    print("--- CONTROLS / KONTROLLER ---")

    for i, name in enumerate(MASA_ROIS_COORDS.keys()):
        print(f"'{i+1}' tuşu: {name}'i düzenleme modunu açar.")
        print(f"'{i+1}' key: Toggles edit mode for {name}.")

    while True:
        # Klavye girişlerini oku
        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Düzenleme modu tuş kontrolleri
        # Edit mode key controls
        key_char = chr(key)
        if key_char.isdigit():
            table_index = int(key_char) - 1
            if 0 <= table_index < len(MASA_ROIS_COORDS):
                table_name = list(MASA_ROIS_COORDS.keys())[table_index]

                if not APP_STATE["is_in_edit_mode"]:
                    print(f"\n!!! DÜZENLEME MODU AKTİF: {table_name} !!!")
                    print(f"!!! EDIT MODE ACTIVE: {table_name} !!!")
                    print(f"Lütfen {POINTS_PER_TABLE} köşe noktasına sırayla tıklayın.")
                    print(f"Please click on the {POINTS_PER_TABLE} corner points sequentially.")
                    APP_STATE["is_in_edit_mode"] = True
                    APP_STATE["editing_table_name"] = table_name
                    APP_STATE["temp_clicks"] = []
                elif APP_STATE["is_in_edit_mode"]:
                    print("Hata: Zaten bir masa düzenleme modundasınız. Lütfen önce bitirin.")
                    print("Error: You are already in edit mode for a desk. Please finish first.")

        # Kameradan yeni bir kare (frame) oku
        # Read a new frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Hata: Görüntü alınamadı.")
            print("Error: Could not read frame from camera.")
            break

        if not APP_STATE["is_in_edit_mode"]:
            # Normal mod: Modeli çalıştır
            # Normal mode: Run the model
            results = model(frame,
                            classes=TARGET_CLASSES,
                            verbose=False,
                            conf=CONF_THRESHOLD)

            # İş mantığını çalıştır ve sonuçları filtrele
            # Run business logic and filter results
            masa_durumlari, inside_indices = check_occupancy(results, MASA_ROIS_NP)
            filtered_results = results[0][inside_indices]

            # Sadece filtreli nesnelerin kutularını çizdir
            # Plot boxes only for filtered results
            annotated_frame = filtered_results.plot(line_width=3, font_size=1.0)

        else:
            # Düzenleme modu: Ham görüntüyü kullan
            # Edit mode: Use the raw frame
            annotated_frame = frame.copy()
            masa_durumlari = {name: "BOS / AVAILABLE" for name in MASA_ROIS_NP.keys()}

        # Masa Durumlarını (ROI) her durumda çiz
        # Draw Table Status (ROI) in all modes
        for name, poly in MASA_ROIS_NP.items():
            if APP_STATE["is_in_edit_mode"] and APP_STATE["editing_table_name"] == name:
                renk = (0, 255, 255) # Sarı / Yellow
                yazi = "DUZENLENIYOR / EDITING..."
            else:
                durum = masa_durumlari.get(name, "BOS / AVAILABLE")
                renk = (0, 0, 255) if durum == "DOLU / OCCUPIED" else (0, 255, 0) # Kırmızı / Yeşil
                yazi = durum

            cv2.polylines(annotated_frame, [poly], isClosed=True, color=renk, thickness=3)
            text_pos = (poly[0][0], poly[0][1] - 10)
            cv2.putText(annotated_frame, f"{name}: {yazi}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, renk, 2, cv2.LINE_AA)

        # Düzenleme modu arayüzünü (banner) çiz
        # Draw the edit mode UI (banner)
        if APP_STATE["is_in_edit_mode"]:
            msg_tr = f"DUZENLENIYOR: {APP_STATE['editing_table_name']} ({len(APP_STATE['temp_clicks'])}/{POINTS_PER_TABLE} Tiklama)"
            msg_en = f"EDITING: {APP_STATE['editing_table_name']} ({len(APP_STATE['temp_clicks'])}/{POINTS_PER_TABLE} Clicks)"

            # Metinleri üst üste iki satır olarak yaz
            # Write the texts as two stacked lines
            cv2.putText(annotated_frame, msg_tr, (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(annotated_frame, msg_en, (20, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

            # Tıklanan noktaları daire ile göster
            # Show clicked points as circles
            for (cx, cy) in APP_STATE["temp_clicks"]:
                cv2.circle(annotated_frame, (cx, cy), 10, (0, 0, 255), 5)

        # Son görüntüyü göster
        # Display the final frame
        cv2.imshow(window_name, annotated_frame)

    # Kaynakları serbest bırak
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
