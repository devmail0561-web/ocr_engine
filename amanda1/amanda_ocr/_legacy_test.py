# Legacy EasyOCR test — not part of the current Tesseract pipeline
import cv2
import warnings

# Supprimer l'avertissement pin_memory de PyTorch (il n'y a pas de GPU disponible)
warnings.filterwarnings('ignore', message="'pin_memory' argument is set as true")

from imgprocessor.text_detection import TextDetector

# Créer le détecteur
detector = TextDetector(engine='easyocr')

# Charger une image
image = cv2.imread('amanda1/image.png')

if image is None:
    print("❌ Impossible de charger l'image")
else:
    print("✅ Image chargée")
    
    # MODE 1: Extraire texte seulement
    try:
        text = detector.extract_text(image, return_text=True, return_coords=False)
        print(f"\n📝 Texte détecté:\n{text}")
    except Exception as e:
        print(f"❌ Erreur extraction texte: {e}")
    
    # MODE 2: Extraire coordonnées seulement
    #try:
        #coords = detector.extract_text(image, return_text=False, return_coords=True)
        #if coords:
            #print(f"\n🎯 Régions détectées: {len(coords)}")
            #for i, region in enumerate(coords):
                #print(f"   Région {i+1}: {region}")
        #else:
            #print("\n⚠️ Aucune région de texte détectée")
    #except Exception as e:
        #print(f"❌ Erreur coords: {e}")
    
    # MODE 3: Extraire texte ET coordonnées
    #try:
        #result = detector.extract_text(image, return_text=True, return_coords=False)
        #if isinstance(result, tuple):
            #text, coords = result
            #print(f"\n📊 Texte + Coordonnées:")
            #print(f"   Texte: {text[:100]}")
            #print(f"   Régions: {len(coords)}")
        #else:
            ##xcept Exception as e:
        #print(f"❌ Erreur mode 3: {e}")