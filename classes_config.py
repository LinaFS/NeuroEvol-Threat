"""
classes_config.py - Configuración de Clases Reales
Basado en tus modelos entrenados con train-GA.py
"""

# ============================================
# CLASES DEL MODELO DE COMPORTAMIENTOS SOSPECHOSOS
# ============================================
BEHAVIOR_CLASSES = {
    0: 'Abuse',
    1: 'Arrest',
    2: 'Arson',
    3: 'Assault',
    4: 'Burglary',
    5: 'Clapping',
    6: 'Explosion',
    7: 'Fighting',
    8: 'Meet_and_Split',
    9: 'Normal_Videos',
    10: 'RoadAccidents',
    11: 'Robbery',
    12: 'Shooting',
    13: 'Shoplifting',
    14: 'Sitting',
    15: 'Standing_Still',
    16: 'Stealing',
    17: 'Vandalism',
    18: 'Walking',
    19: 'Walking_While_Reading_Book',
    20: 'Walking_While_Using_Phone'
}

# ============================================
# CLASES DEL MODELO DE ARMAS
# ============================================
WEAPON_CLASSES = {
    0: 'Armas de Fuego',
    1: 'Contundentes',
    2: 'De control',
    3: 'Explosivos',
    4: 'Punzocortante',
    5: 'Restriccion'
}

# ============================================
# NIVELES DE RIESGO POR CLASE DE COMPORTAMIENTO
# ============================================
CLASS_RISK_LEVELS = {
    # ===== ALTO RIESGO (3) - Requiere acción inmediata =====
    0: 3,   # Abuse - Abuso/Maltrato
    2: 3,   # Arson - Incendio provocado
    3: 3,   # Assault - Asalto/Agresión
    6: 3,   # Explosion - Explosión
    7: 3,   # Fighting - Pelea/Riña
    11: 3,  # Robbery - Robo con violencia
    12: 3,  # Shooting - Tiroteo
    
    # ===== MEDIO RIESGO (2) - Requiere atención =====
    1: 2,   # Arrest - Arresto (puede ser confuso)
    4: 2,   # Burglary - Robo con allanamiento
    10: 2,  # RoadAccidents - Accidentes viales
    13: 2,  # Shoplifting - Hurto en tiendas
    16: 2,  # Stealing - Robo/Hurto
    17: 2,  # Vandalism - Vandalismo
    
    # ===== BAJO RIESGO (1) - Monitoreo rutinario =====
    8: 1,   # Meet_and_Split - Encuentro y separación
    14: 1,  # Sitting - Sentado (permanencia prolongada)
    15: 1,  # Standing_Still - De pie inmóvil (sospechoso si es prolongado)
    
    # ===== SIN RIESGO (0) - Comportamiento normal =====
    5: 0,   # Clapping - Aplaudiendo
    9: 0,   # Normal_Videos - Videos normales
    18: 0,  # Walking - Caminando
    19: 0,  # Walking_While_Reading_Book - Caminando leyendo
    20: 0,  # Walking_While_Using_Phone - Caminando usando teléfono
}

# ============================================
# NIVELES DE RIESGO POR CLASE DE ARMA
# ============================================
WEAPON_RISK_LEVELS = {
    0: 3,  # Armas de Fuego - ALTO
    1: 2,  # Contundentes - MEDIO
    2: 2,  # De control - MEDIO
    3: 3,  # Explosivos - ALTO
    4: 3,  # Punzocortante - ALTO
    5: 2,  # Restriccion - MEDIO
}

# ============================================
# COLORES PARA VISUALIZACIÓN (BGR)
# ============================================
BEHAVIOR_COLORS = {
    # Alto riesgo - Rojo
    'Abuse': (0, 0, 255),
    'Arson': (0, 0, 200),
    'Assault': (0, 0, 255),
    'Explosion': (0, 0, 128),
    'Fighting': (0, 0, 255),
    'Robbery': (0, 0, 200),
    'Shooting': (0, 0, 128),
    
    # Medio riesgo - Naranja
    'Arrest': (0, 165, 255),
    'Burglary': (0, 140, 255),
    'RoadAccidents': (0, 200, 255),
    'Shoplifting': (0, 165, 255),
    'Stealing': (0, 140, 255),
    'Vandalism': (0, 165, 255),
    
    # Bajo riesgo - Amarillo
    'Meet_and_Split': (0, 255, 255),
    'Sitting': (0, 255, 200),
    'Standing_Still': (0, 255, 200),
    
    # Sin riesgo - Verde
    'Clapping': (0, 255, 0),
    'Normal_Videos': (0, 200, 0),
    'Walking': (0, 255, 0),
    'Walking_While_Reading_Book': (0, 200, 0),
    'Walking_While_Using_Phone': (0, 200, 0),
}

WEAPON_COLORS = {
    'Armas de Fuego': (0, 0, 128),      # Rojo oscuro
    'Contundentes': (0, 165, 255),      # Naranja
    'De control': (0, 165, 255),        # Naranja
    'Explosivos': (0, 0, 128),          # Rojo oscuro
    'Punzocortante': (0, 0, 255),       # Rojo
    'Restriccion': (0, 165, 255),       # Naranja
}

# ============================================
# DESCRIPCIONES DE CLASES
# ============================================
CLASS_DESCRIPTIONS = {
    # Comportamientos
    'Abuse': 'Abuso físico o maltrato',
    'Arrest': 'Arresto o detención',
    'Arson': 'Incendio provocado',
    'Assault': 'Agresión o asalto',
    'Burglary': 'Robo con allanamiento',
    'Clapping': 'Aplaudiendo (evento normal)',
    'Explosion': 'Explosión o detonación',
    'Fighting': 'Pelea o riña violenta',
    'Meet_and_Split': 'Encuentro y separación de personas',
    'Normal_Videos': 'Actividad completamente normal',
    'RoadAccidents': 'Accidente de tránsito',
    'Robbery': 'Robo con violencia',
    'Shooting': 'Tiroteo o disparos',
    'Shoplifting': 'Hurto en establecimiento comercial',
    'Sitting': 'Persona sentada (posible loitering)',
    'Standing_Still': 'Persona inmóvil (posible loitering)',
    'Stealing': 'Robo o hurto',
    'Vandalism': 'Vandalismo o destrucción de propiedad',
    'Walking': 'Caminando normalmente',
    'Walking_While_Reading_Book': 'Caminando mientras lee',
    'Walking_While_Using_Phone': 'Caminando usando teléfono',
    
    # Armas
    'Armas de Fuego': 'Pistolas, rifles, escopetas',
    'Contundentes': 'Bates, palos, objetos contundentes',
    'De control': 'Gas pimienta, tasers',
    'Explosivos': 'Bombas, granadas, material explosivo',
    'Punzocortante': 'Cuchillos, navajas, objetos punzantes',
    'Restriccion': 'Esposas, restricciones físicas'
}

# ============================================
# MAPEO DE CLASES A CATEGORÍAS DE ANÁLISIS TEMPORAL
# ============================================
# Estas categorías ayudan al análisis temporal a decidir qué métricas priorizar

TEMPORAL_CATEGORIES = {
    # Clases que requieren análisis de velocidad
    'high_velocity': ['Assault', 'Fighting', 'Robbery', 'RoadAccidents'],
    
    # Clases que requieren análisis de permanencia (loitering)
    'dwelling_analysis': ['Sitting', 'Standing_Still', 'Meet_and_Split', 'Shoplifting'],
    
    # Clases que requieren análisis de trayectoria errática
    'erratic_movement': ['Assault', 'Fighting', 'Burglary', 'Stealing'],
    
    # Clases normales (no generan alertas)
    'normal': ['Normal_Videos', 'Walking', 'Clapping', 'Walking_While_Reading_Book', 
               'Walking_While_Using_Phone'],
    
    # Clases que siempre son críticas (alerta inmediata)
    'critical': ['Shooting', 'Explosion', 'Arson', 'Abuse'],
}

# ============================================
# CONFIGURACIÓN DE UMBRALES POR CATEGORÍA
# ============================================
THRESHOLDS_BY_CATEGORY = {
    'high_velocity': {
        'velocity_min': 8.0,        # px/frame mínimo para considerar "alta velocidad"
        'acceleration_min': 5.0,    # Aceleración mínima
        'direction_changes': 5,     # Cambios de dirección
    },
    'dwelling_analysis': {
        'dwelling_time_min': 5.0,   # Segundos mínimos de permanencia
        'velocity_max': 1.0,        # Velocidad máxima para considerar "estático"
        'area_threshold': 0.02,     # Área mínima cubierta
    },
    'erratic_movement': {
        'direction_changes_min': 7, # Cambios de dirección mínimos
        'curvature_min': 0.7,       # Curvatura de trayectoria
        'velocity_std_min': 3.0,    # Desviación estándar de velocidad
    },
    'critical': {
        # Alertas inmediatas sin análisis temporal profundo
        'instant_alert': True,
        'alert_level': 3,
    }
}

# ============================================
# FUNCIÓN AUXILIAR: OBTENER NIVEL DE RIESGO
# ============================================
def get_risk_level(class_id, is_weapon=False):
    """
    Obtiene el nivel de riesgo de una clase
    
    Args:
        class_id: ID de la clase
        is_weapon: True si es del modelo de armas
    
    Returns:
        int: Nivel de riesgo (0-3)
    """
    if is_weapon:
        return WEAPON_RISK_LEVELS.get(class_id, 1)
    else:
        return CLASS_RISK_LEVELS.get(class_id, 1)

def get_class_name(class_id, is_weapon=False):
    """
    Obtiene el nombre de una clase
    
    Args:
        class_id: ID de la clase
        is_weapon: True si es del modelo de armas
    
    Returns:
        str: Nombre de la clase
    """
    if is_weapon:
        return WEAPON_CLASSES.get(class_id, 'Unknown')
    else:
        return BEHAVIOR_CLASSES.get(class_id, 'Unknown')

def get_class_color(class_name, is_weapon=False):
    """
    Obtiene el color BGR para visualización
    
    Args:
        class_name: Nombre de la clase
        is_weapon: True si es del modelo de armas
    
    Returns:
        tuple: Color en formato BGR
    """
    if is_weapon:
        return WEAPON_COLORS.get(class_name, (255, 255, 255))
    else:
        return BEHAVIOR_COLORS.get(class_name, (255, 255, 255))

def get_temporal_category(class_name):
    """
    Obtiene la categoría de análisis temporal
    
    Args:
        class_name: Nombre de la clase
    
    Returns:
        str: Categoría ('high_velocity', 'dwelling_analysis', etc.)
    """
    for category, classes in TEMPORAL_CATEGORIES.items():
        if class_name in classes:
            return category
    return 'normal'

def should_analyze_temporally(class_name):
    """
    Determina si una clase requiere análisis temporal profundo
    
    Args:
        class_name: Nombre de la clase
    
    Returns:
        bool: True si requiere análisis temporal
    """
    category = get_temporal_category(class_name)
    return category not in ['normal', 'critical']

# ============================================
# INFORMACIÓN DEL SISTEMA
# ============================================
def print_classes_summary():
    """Imprime un resumen de las clases configuradas"""
    print("\n" + "="*70)
    print("📊 CONFIGURACIÓN DE CLASES - NEUROEVOL-THREAT")
    print("="*70)
    
    print(f"\n🎭 Clases de Comportamiento: {len(BEHAVIOR_CLASSES)}")
    print(f"🔫 Clases de Armas: {len(WEAPON_CLASSES)}")
    
    print("\n📈 Distribución por nivel de riesgo (Comportamientos):")
    risk_dist = {0: [], 1: [], 2: [], 3: []}
    for class_id, risk in CLASS_RISK_LEVELS.items():
        risk_dist[risk].append(BEHAVIOR_CLASSES[class_id])
    
    for level in [3, 2, 1, 0]:
        emoji = ['⚪', '🟡', '🟠', '🔴'][level]
        level_name = ['Sin riesgo', 'Bajo', 'Medio', 'Alto'][level]
        classes = risk_dist[level]
        print(f"   {emoji} {level_name:12s}: {len(classes):2d} clases - {', '.join(classes[:3])}" + 
              (f"..." if len(classes) > 3 else ""))
    
    print("\n🔫 Distribución por nivel de riesgo (Armas):")
    weapon_risk_dist = {0: [], 1: [], 2: [], 3: []}
    for class_id, risk in WEAPON_RISK_LEVELS.items():
        weapon_risk_dist[risk].append(WEAPON_CLASSES[class_id])
    
    for level in [3, 2, 1, 0]:
        if weapon_risk_dist[level]:
            emoji = ['⚪', '🟡', '🟠', '🔴'][level]
            level_name = ['Sin riesgo', 'Bajo', 'Medio', 'Alto'][level]
            classes = weapon_risk_dist[level]
            print(f"   {emoji} {level_name:12s}: {len(classes):2d} clases - {', '.join(classes)}")
    
    print("\n" + "="*70)

# ============================================
# TEST DE CONFIGURACIÓN
# ============================================
if __name__ == "__main__":
    print_classes_summary()
    
    # Pruebas
    print("\n🧪 Pruebas de funciones:")
    
    test_cases = [
        (3, False, "Assault"),
        (12, False, "Shooting"),
        (9, False, "Normal_Videos"),
        (0, True, "Armas de Fuego"),
    ]
    
    for class_id, is_weapon, expected_name in test_cases:
        name = get_class_name(class_id, is_weapon)
        risk = get_risk_level(class_id, is_weapon)
        category = get_temporal_category(name) if not is_weapon else 'weapon'
        
        print(f"\n   Clase {class_id} ({'Arma' if is_weapon else 'Comportamiento'}):")
        print(f"   - Nombre: {name}")
        print(f"   - Riesgo: {risk}")
        print(f"   - Categoría: {category}")
        print(f"   - Requiere análisis temporal: {should_analyze_temporally(name)}")