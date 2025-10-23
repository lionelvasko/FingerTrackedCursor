# FingerTrackedCursor
FingerTrackedCursor is a computer vision project from the MSc Software Engineering program at the University of Szeged. It enables mouse control via finger tracking using a webcam. Two implementations exist: one with Google’s MediaPipe, and one with a custom Python-based solution.


## Áttekintés

Ez a projekt lehetővé teszi az egérkurzor vezérlését webkamera és képfeldolgozás segítségével. Két megvalósítást tartalmaz a repository:

- `mediapipe-version/` – Google MediaPipe alapú megoldás (már létező fájlokkal a mappában).
- `opencv-version/` – egyszerű, OpenCV alapú implementáció (contour / fingertip alapú detektálás).

Mindkét implementáció a kéz ujjainak helyzetét felhasználva tudja a kurzort mozgatni és alapműveleteket (bal/kattintás, jobb kattintás, görgetés) végrehajtani.

## Repository szerkezet

Főbb fájlok és mappák (releváns részletek):

- `mediapipe-version/`
	- `main.py` – MediaPipe alapú futtatható demo.
	- `actions.py` – Az egérműveletek (pyautogui) implementációi.
	- `gestures.py` – Gesztusdetektálás MediaPipe landmarkok alapján.
	- `gesture-map.json` – Gesztus -> akció leképezés.
	- `resources/hand-landmarker.task` – MediaPipe modell (nincs benne a README-ben, a kódban hivatkozva).

- `opencv-version/` (újonnan hozzáadva)
	- `main.py` – OpenCV alapú futtatható demo (webkamera olvasás, ujjhegy detektálás, gesztus logika).
	- `actions.py` – Ugyanaz a pyautogui-alapú akciók (move, left_click, right_click, scroll_up/down).
	- `gestures.py` – Egyszerű heuristikák gesztusok felismerésére kontúrok és ujjhegyek alapján.
	- `gesture-map.json` – Gesztus -> akció leképezés (azonos a `mediapipe-version`-nel).
	- `requirements.txt` – futtatáshoz szükséges Python csomagok listája.

## Futtatási útmutató (Windows)

1. Hozz létre egy virtuális környezetet (ajánlott):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. MediaPipe verzió futtatása

```powershell
cd mediapipe-version
pip install -r requirements.txt
python main.py
```

Megjegyzés: A MediaPipe telepítése platformfüggő lehet. Ha telepítési problémák merülnek fel, nézd meg a MediaPipe hivatalos dokumentációját.

3. OpenCV verzió futtatása

```powershell
cd opencv-version
pip install -r requirements.txt
python main.py
```

4. Kilépés: az alkalmazásablakban nyomd meg az ESC-et.

## Tervezett viselkedés és gesztusok

- `pinch` – bal kattintás
- `open_palm` – kurzormozgatás (index ujj hegyét követi)
- `fist` – jobb kattintás
- `two_fingers` – görgetés fel
- `thumbs_up` – görgetés le

Mindkét implementáció ugyanazt a `gesture-map.json`-t használja a leképezéshez.

## Biztonság és engedélyek

- `pyautogui`-t használjuk az egérműveletekhez; a Windows rendszeren engedélyezni kell a képernyőfelvételt és az input-vezérlést, ha a rendszer blokkolja.

## További teendők / javaslatok

- Finomhangolni a gesztusérzékenységet mindkét implementációnál.
- Hozzáadni konfigurációs fájlt (pl. `config.yaml`) a küszöbértékek és a kamera index beállításához.
- Tesztek hozzáadása a gesztusdetektorokhoz (unit tesztek heuristikákra).

---

Ha szeretnéd, most lefuttatom a változtatásokat, vagy létrehozom az `opencv-version/` fájlokat a munkakönyvtárban. Mit szeretnél, folytassam azonnal az implementációval?
