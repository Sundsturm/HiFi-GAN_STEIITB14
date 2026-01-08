#!/bin/bash

# ==============================================================================
# HIFI-GAN INTEGRATION SCRIPT (FILE EXISTING)
# ==============================================================================
# 1. Menyiapkan folder rtl/generator
# 2. Memindahkan file .v yang sudah ada ke folder tersebut
# 3. Membuat script TCL untuk Vivado
# ==============================================================================

# 1. SETUP DIREKTORI
echo "[INFO] Memastikan folder tujuan ada..."
mkdir -p rtl/generator

# 2. PINDAHKAN FILE (Organizing)
# Kita pindahkan file hanya jika file tersebut ada di folder saat ini (root)
echo "[INFO] Memindahkan file ke rtl/generator..."

# Daftar file yang harus dipindahkan
files=("mrf_block.v" "residual_block.v" "upsample_module.v" "generator_top.v")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo " -> Memindahkan $file ke rtl/generator/"
        mv "$file" rtl/generator/
    elif [ -f "rtl/generator/$file" ]; then
        echo " -> $file sudah berada di rtl/generator/ (Aman)"
    else
        echo " [WARNING] File $file tidak ditemukan di folder ini! Pastikan nama file sesuai."
    fi
done

# 3. BUAT SCRIPT KONEKSI VIVADO (.tcl)
echo "[INFO] Membuat script TCL (add_hifigan_sources.tcl)..."

cat << 'EOF' > add_hifigan_sources.tcl
# =========================================================
# TCL Script untuk Vivado
# Cara pakai: Ketik "source add_hifigan_sources.tcl" di Tcl Console
# =========================================================

# 1. Hapus referensi file lama yang mungkin broken (Opsional, agar bersih)
# remove_files [get_files rtl/generator/*.v] 

# 2. Add Shared Modules (Folder 'shared' dan subfoldernya)
# Pastikan folder rtl/shared ada. Jika folder shared Anda ada di root, ubah path di bawah.
if {[file exists rtl/shared]} {
    add_files [glob -nocomplain rtl/shared/*/*.v]
    add_files [glob -nocomplain rtl/shared/*.v]
} else {
    puts " [INFO] Folder rtl/shared tidak terdeteksi script, pastikan Anda menambahkannya manual jika perlu."
}

# 3. Add Generator Modules (Folder yang baru kita rapikan)
add_files rtl/generator/upsample_module.v
add_files rtl/generator/residual_block.v
add_files rtl/generator/mrf_block.v
add_files rtl/generator/generator_top.v

# 4. Refresh Hierarchy
update_compile_order -fileset sources_1

puts "-------------------------------------------------------"
puts " SUCCESS: Integrasi selesai."
puts " Silakan cek panel 'Sources' -> 'Hierarchy'."
puts "-------------------------------------------------------"
EOF

echo "---------------------------------------------------------"
echo "SELESAI!"
echo "1. File-file Anda sekarang seharusnya ada di: rtl/generator/"
echo "2. Jalankan perintah ini di Vivado Tcl Console:"
echo "   source add_hifigan_sources.tcl"
echo "---------------------------------------------------------"