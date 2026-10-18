# Puzzle Cam — Hand Frame Capture

Struktur folder:

    index.html            kerangka halaman
    css/styles.css        tampilan
    app.js                kamera + MediaPipe + puzzle
    hand_landmarker.task  model tangan (opsional, kalau tidak ada diunduh otomatis)

Cara jalan: klik kanan index.html -> Open with Live Server, lalu izinkan kamera.

Cara main:
1. Munculkan dua tangan, bingkai kuning muncul di antara titik jempol-telunjuk.
2. Cubit kedua tangan & tahan ~0.7 detik -> foto diambil dan dipecah jadi puzzle.
3. Cubit potongan, geser, lepas di potongan lain untuk menukar.
Tombol: Space = ambil foto manual, R = ulang, +/- = ukuran puzzle.
