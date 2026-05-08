#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# Wczytanie modelu bazowego
# ============================================================

def wczytaj_model(sciezka_modelu: str):
    try:
        model = YOLO(sciezka_modelu)
        print(f"[OK] Wczytano model: {sciezka_modelu}")
        return model
    except Exception as blad:
        raise RuntimeError(f"Nie udało się wczytać modelu '{sciezka_modelu}': {blad}") from blad


# ============================================================
# Wczytanie obrazu
# ============================================================

def wczytaj_obraz(sciezka_obrazu: str) -> np.ndarray:
    obraz = cv2.imread(sciezka_obrazu)
    if obraz is None:
        raise FileNotFoundError(f"Błąd: nie można wczytać obrazu: {sciezka_obrazu}")
    return obraz


# ============================================================
# Wczytanie strumienia wideo
# ============================================================

def wczytaj_strumien(zrodlo: str):
    if zrodlo == "camera":
        cap = cv2.VideoCapture(0)
        opis = "kamera domyślna"
    else:
        cap = cv2.VideoCapture(zrodlo)
        opis = zrodlo

    if not cap.isOpened():
        raise RuntimeError(f"Błąd: nie można otworzyć źródła wideo: {opis}")

    return cap


# ============================================================
# Uruchomienie detekcji
# ============================================================

def wykonaj_detekcje(model, obraz: np.ndarray):
    return model(obraz, verbose=False)


# ============================================================
# Odczyt danych z wyników YOLO
# ============================================================

def odczytaj_wyniki(wyniki) -> List[Dict[str, Any]]:
    detekcje: List[Dict[str, Any]] = []

    if wyniki is None or len(wyniki) == 0:
        return detekcje

    wynik = wyniki[0]
    boxes = getattr(wynik, "boxes", None)
    if boxes is None:
        return detekcje

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())

        detekcje.append({
            "box": [float(x) for x in xyxy],
            "class_id": cls,
            "confidence": conf,
        })

    return detekcje


# ============================================================
# Filtrowanie detekcji
# ============================================================

def filtruj_detekcje(
    detekcje: List[Dict[str, Any]],
    prog_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    return [det for det in detekcje if det["confidence"] >= prog_confidence]


# ============================================================
# Pobranie nazw klas
# ============================================================

def pobierz_nazwy_klas(model) -> Optional[Dict[int, str]]:
    names = getattr(model, "names", None)
    if names is None:
        return None

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, list):
        return {i: str(nazwa) for i, nazwa in enumerate(names)}

    return None


# ============================================================
# Rysowanie detekcji
# ============================================================

def rysuj_detekcje(
    obraz: np.ndarray,
    detekcje: List[Dict[str, Any]],
    nazwy_klas: Optional[Dict[int, str]] = None
) -> np.ndarray:
    wynik = obraz.copy()

    for det in detekcje:
        x1, y1, x2, y2 = map(int, det["box"])
        class_id = int(det["class_id"])
        confidence = float(det["confidence"])

        if nazwy_klas is not None and class_id in nazwy_klas:
            label = nazwy_klas[class_id]
        else:
            label = f"class_{class_id}"

        tekst = f"{label}: {confidence:.2f}"

        cv2.rectangle(wynik, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (szer_tekstu, wys_tekstu), _ = cv2.getTextSize(
            tekst,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )
        y_tekstu = max(y1 - 10, wys_tekstu + 10)
        cv2.rectangle(
            wynik,
            (x1, y_tekstu - wys_tekstu - 6),
            (x1 + szer_tekstu + 6, y_tekstu + 4),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            wynik,
            tekst,
            (x1 + 3, y_tekstu),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    return wynik


# ============================================================
# Napisy diagnostyczne
# ============================================================

def dodaj_diagnostyke(
    obraz: np.ndarray,
    liczba_detekcji: int,
    prog_confidence: float
) -> np.ndarray:
    wynik = obraz.copy()
    tekst1 = f"Detekcje: {liczba_detekcji}"
    tekst2 = f"Prog confidence: {prog_confidence:.2f}"

    cv2.putText(wynik, tekst1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(wynik, tekst2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return wynik


# ============================================================
# Sprawdzenie struktury danych treningowych
# ============================================================

def sprawdz_strukture_danych(sciezka_danych: str) -> bool:
    wymagane_katalogi = [
        os.path.join(sciezka_danych, "images", "train"),
        os.path.join(sciezka_danych, "images", "val"),
        os.path.join(sciezka_danych, "labels", "train"),
        os.path.join(sciezka_danych, "labels", "val"),
    ]
    wymagany_yaml = os.path.join(sciezka_danych, "data.yaml")

    wszystko_ok = True

    if not os.path.isdir(sciezka_danych):
        print(f"[BŁĄD] Nie istnieje katalog datasetu: {sciezka_danych}")
        return False

    for katalog in wymagane_katalogi:
        if not os.path.isdir(katalog):
            print(f"[BŁĄD] Brak katalogu: {katalog}")
            wszystko_ok = False

    if not os.path.isfile(wymagany_yaml):
        print(f"[BŁĄD] Brak pliku: {wymagany_yaml}")
        wszystko_ok = False

    if wszystko_ok:
        print("[OK] Struktura danych treningowych wygląda poprawnie.")

    return wszystko_ok

# ============================================================
# Dotrenowanie modelu
# ============================================================

def dotrenuj_model(
    model,
    sciezka_danych: str,
    liczba_epok: int = 20,
    rozmiar_obrazu: int = 640
):
    if not sprawdz_strukture_danych(sciezka_danych):
        raise RuntimeError("Niepoprawna struktura danych treningowych. Trening przerwany.")

    sciezka_yaml = os.path.join(sciezka_danych, "data.yaml")
    print("[INFO] Rozpoczynam trening modelu...")
    print(f"[INFO] data={sciezka_yaml}, epochs={liczba_epok}, imgsz={rozmiar_obrazu}")

    wyniki_treningu = model.train(
        data=sciezka_yaml,
        epochs=liczba_epok,
        imgsz=rozmiar_obrazu
    )

    print("[OK] Trening zakończony.")
    print("[INFO] Najlepszy model zwykle znajduje się w: runs/detect/train/weights/best.pt")
    return wyniki_treningu


# ============================================================
# Wczytanie modelu dotrenowanego
# ============================================================

def wczytaj_model_dotrenowany(sciezka_modelu: str):
    return wczytaj_model(sciezka_modelu)


# ============================================================
# Porównanie modeli
# ============================================================

def porownaj_modele(
    model_bazowy,
    model_dotrenowany,
    obraz: np.ndarray,
    prog_confidence: float
) -> Tuple[np.ndarray, np.ndarray]:
    wyniki_bazowe = wykonaj_detekcje(model_bazowy, obraz)
    detekcje_bazowe = odczytaj_wyniki(wyniki_bazowe)
    detekcje_bazowe = filtruj_detekcje(detekcje_bazowe, prog_confidence)

    wyniki_dotrenowane = wykonaj_detekcje(model_dotrenowany, obraz)
    detekcje_dotrenowane = odczytaj_wyniki(wyniki_dotrenowane)
    detekcje_dotrenowane = filtruj_detekcje(detekcje_dotrenowane, prog_confidence)

    nazwy_bazowe = pobierz_nazwy_klas(model_bazowy)
    nazwy_dotrenowane = pobierz_nazwy_klas(model_dotrenowany)

    obraz_bazowy = rysuj_detekcje(obraz, detekcje_bazowe, nazwy_bazowe)
    obraz_bazowy = dodaj_diagnostyke(obraz_bazowy, len(detekcje_bazowe), prog_confidence)
    cv2.putText(obraz_bazowy, "MODEL BAZOWY", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    obraz_dotrenowany = rysuj_detekcje(obraz, detekcje_dotrenowane, nazwy_dotrenowane)
    obraz_dotrenowany = dodaj_diagnostyke(obraz_dotrenowany, len(detekcje_dotrenowane), prog_confidence)
    cv2.putText(obraz_dotrenowany, "MODEL DOTRENOWANY", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    print(f"[INFO] Model bazowy - liczba detekcji: {len(detekcje_bazowe)}")
    print(f"[INFO] Model dotrenowany - liczba detekcji: {len(detekcje_dotrenowane)}")

    return obraz_bazowy, obraz_dotrenowany


# ============================================================
# Przetwarzanie obrazu
# ============================================================

def przetwarzaj_obraz(model, sciezka_obrazu: str, prog_confidence: float):
    obraz = wczytaj_obraz(sciezka_obrazu)

    wyniki = wykonaj_detekcje(model, obraz)
    detekcje = odczytaj_wyniki(wyniki)
    detekcje = filtruj_detekcje(detekcje, prog_confidence)

    nazwy_klas = pobierz_nazwy_klas(model)

    obraz_wynikowy = rysuj_detekcje(obraz, detekcje, nazwy_klas)
    obraz_wynikowy = dodaj_diagnostyke(obraz_wynikowy, len(detekcje), prog_confidence)

    print(f"[INFO] Liczba detekcji po filtrowaniu: {len(detekcje)}")
    cv2.imshow("Wynik detekcji - obraz", obraz_wynikowy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Przetwarzanie wideo / kamery
# ============================================================

def przetwarzaj_wideo(model, zrodlo: str, prog_confidence: float):
    """Przetwarza wideo lub obraz z kamery."""
    cap = wczytaj_strumien(zrodlo)
    nazwy_klas = pobierz_nazwy_klas(model)

    while True:
        poprawnie, klatka = cap.read()
        if not poprawnie:
            break

        wyniki = wykonaj_detekcje(model, klatka)
        detekcje = odczytaj_wyniki(wyniki)
        detekcje = filtruj_detekcje(detekcje, prog_confidence)

        klatka_wynikowa = rysuj_detekcje(klatka, detekcje, nazwy_klas)
        klatka_wynikowa = dodaj_diagnostyke(klatka_wynikowa, len(detekcje), prog_confidence)

        cv2.imshow("Wynik detekcji - wideo", klatka_wynikowa)

        klawisz = cv2.waitKey(1) & 0xFF
        if klawisz in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Tryb porównawczy
# ============================================================

def uruchom_porownanie(
    model_bazowy,
    model_dotrenowany,
    sciezka_obrazu: str,
    prog_confidence: float
):
    obraz = wczytaj_obraz(sciezka_obrazu)

    obraz_bazowy, obraz_dotrenowany = porownaj_modele(
        model_bazowy,
        model_dotrenowany,
        obraz,
        prog_confidence
    )

    cv2.imshow("Model bazowy", obraz_bazowy)
    cv2.imshow("Model dotrenowany", obraz_dotrenowany)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Funkcja główna
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LAB6 - YOLO, gotowe rozwiązanie")
    parser.add_argument("--model", required=True, help="Ścieżka do modelu bazowego, np. yolov8n.pt")
    parser.add_argument("--image", help="Ścieżka do obrazu")
    parser.add_argument("--video", help="Ścieżka do pliku wideo")
    parser.add_argument("--camera", action="store_true", help="Użyj kamery")
    parser.add_argument("--confidence", type=float, default=0.5, help="Próg confidence")

    parser.add_argument("--train", action="store_true", help="Uruchom dotrenowanie modelu")
    parser.add_argument("--train-data", help="Ścieżka do danych treningowych YOLO")
    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok treningu")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozmiar obrazu do treningu")
    parser.add_argument("--trained-model", help="Ścieżka do modelu dotrenowanego")
    parser.add_argument("--compare", action="store_true", help="Porównaj model bazowy i dotrenowany")
    parser.add_argument("--show-annotation-help", action="store_true", help="Pokaż informacje o anotacji danych")

    args = parser.parse_args()

    try:
        model_bazowy = wczytaj_model(args.model)

        if args.train:
            if not args.train_data:
                raise ValueError("Dla trybu --train trzeba podać --train-data dataset")
            dotrenuj_model(model_bazowy, args.train_data, args.epochs, args.imgsz)
            return 0

        if args.compare:
            if not args.trained_model:
                raise ValueError("Dla trybu --compare trzeba podać --trained-model")
            if not args.image:
                raise ValueError("Dla trybu --compare trzeba podać --image")

            model_dotrenowany = wczytaj_model_dotrenowany(args.trained_model)
            uruchom_porownanie(model_bazowy, model_dotrenowany, args.image, args.confidence)
            return 0

        if args.image:
            przetwarzaj_obraz(model_bazowy, args.image, args.confidence)
            return 0

        if args.video:
            przetwarzaj_wideo(model_bazowy, args.video, args.confidence)
            return 0

        if args.camera:
            przetwarzaj_wideo(model_bazowy, "camera", args.confidence)
            return 0

        print("Nie podano trybu pracy.")
        print("Użyj jednego z argumentów: --image, --video, --camera, --train albo --compare")
        parser.print_help()
        return 1

    except Exception as blad:
        print(f"[BŁĄD] {blad}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
