import csv
import re
from collections import Counter

def corregir_clase(clase_actual):
    """
    Limpia y corrige el nombre de la clase eliminando sufijos problemáticos.
    
    Reglas:
    1. Elimina paréntesis con números: (73), (87), etc.
    2. Elimina guiones bajos finales
    3. Elimina sufijos técnicos: _x264, _frames, _video
    """
    clase_limpia = clase_actual
    
    # Eliminar paréntesis con números al final: (73), (87), etc.
    clase_limpia = re.sub(r'_?\(\d+\)$', '', clase_limpia)
    
    # Eliminar sufijos técnicos
    clase_limpia = re.sub(r'_x264.*$', '', clase_limpia)
    clase_limpia = re.sub(r'_frames.*$', '', clase_limpia)
    clase_limpia = re.sub(r'_video.*$', '', clase_limpia)
    
    # Eliminar guiones bajos finales
    clase_limpia = clase_limpia.rstrip('_')
    
    # Si quedó vacío, devolver la original
    if not clase_limpia:
        clase_limpia = clase_actual
    
    return clase_limpia

def analizar_csv(ruta_csv):
    """Analiza el CSV y muestra estadísticas de las clases"""
    clases_originales = []
    clases_corregidas = []
    
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clase_original = row['class']
            clase_corregida = corregir_clase(clase_original)
            clases_originales.append(clase_original)
            clases_corregidas.append(clase_corregida)
    
    return clases_originales, clases_corregidas

def mostrar_cambios(clases_originales, clases_corregidas):
    """Muestra un resumen de los cambios que se harán"""
    cambios = {}
    for orig, corr in zip(clases_originales, clases_corregidas):
        if orig != corr:
            if orig not in cambios:
                cambios[orig] = corr
    
    if cambios:
        print("=" * 80)
        print("CAMBIOS QUE SE REALIZARÁN:")
        print("=" * 80)
        for orig, corr in sorted(cambios.items()):
            print(f"  {orig:50s} → {corr}")
        print("=" * 80)
        print(f"Total de clases a modificar: {len(cambios)}")
    else:
        print("✅ No se detectaron clases que necesiten corrección")
    
    return cambios

def corregir_csv(ruta_entrada, ruta_salida):
    """Corrige las clases en el CSV y guarda el resultado"""
    filas_corregidas = []
    total_filas = 0
    filas_modificadas = 0
    
    with open(ruta_entrada, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_filas += 1
            clase_original = row['class']
            clase_corregida = corregir_clase(clase_original)
            
            if clase_original != clase_corregida:
                filas_modificadas += 1
            
            row['class'] = clase_corregida
            filas_corregidas.append(row)
    
    # Guardar CSV corregido
    with open(ruta_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filas_corregidas)
    
    return total_filas, filas_modificadas

def generar_estadisticas(ruta_csv):
    """Genera estadísticas de las clases corregidas"""
    clases = []
    
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clases.append(row['class'])
    
    contador = Counter(clases)
    
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DE CLASES (DESPUÉS DE CORRECCIÓN):")
    print("=" * 80)
    print(f"  {'CLASE':40s} | {'DETECCIONES':>12s}")
    print("-" * 80)
    
    for clase, count in sorted(contador.items()):
        print(f"  {clase:40s} | {count:12d}")
    
    print("-" * 80)
    print(f"  {'TOTAL':40s} | {sum(contador.values()):12d}")
    print(f"  {'CLASES ÚNICAS':40s} | {len(contador):12d}")
    print("=" * 80)

# ==================== EJECUCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    # Configuración
    RUTA_CSV_ORIGINAL = 'resultados/anotaciones.csv'
    RUTA_CSV_CORREGIDO = 'resultados/anotaciones_corregido.csv'
    
    print("🔧 CORRECTOR DE CLASES EN CSV")
    print("=" * 80)
    
    try:
        # 1. Analizar el CSV original
        print("\n📊 Analizando CSV original...")
        clases_orig, clases_corr = analizar_csv(RUTA_CSV_ORIGINAL)
        
        # 2. Mostrar los cambios que se realizarán
        print("\n🔍 Detectando cambios necesarios...")
        cambios = mostrar_cambios(clases_orig, clases_corr)
        
        if not cambios:
            print("\n✅ El CSV no necesita correcciones")
            exit(0)
        
        # 3. Pedir confirmación
        print("\n⚠️  ¿Deseas proceder con la corrección? (s/n): ", end="")
        respuesta = input().strip().lower()
        
        if respuesta != 's':
            print("❌ Operación cancelada")
            exit(0)
        
        # 4. Corregir el CSV
        print("\n🔄 Corrigiendo CSV...")
        total, modificadas = corregir_csv(RUTA_CSV_ORIGINAL, RUTA_CSV_CORREGIDO)
        
        print(f"\n✅ CSV corregido guardado en: {RUTA_CSV_CORREGIDO}")
        print(f"   Total de filas: {total}")
        print(f"   Filas modificadas: {modificadas}")
        
        # 5. Generar estadísticas del CSV corregido
        generar_estadisticas(RUTA_CSV_CORREGIDO)
        
        print("\n💡 SIGUIENTE PASO:")
        print(f"   Si todo se ve bien, puedes reemplazar el original:")
        print(f"   mv {RUTA_CSV_CORREGIDO} {RUTA_CSV_ORIGINAL}")
        
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el archivo '{RUTA_CSV_ORIGINAL}'")
        print("   Verifica que la ruta sea correcta")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")