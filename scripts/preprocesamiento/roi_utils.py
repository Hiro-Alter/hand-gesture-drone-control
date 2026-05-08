# ================================
#  Utilidades de ROI
# ================================

def make_square_bbox(x, y, w, h, img_w, img_h):
    """
    Convierte un bbox rectangular a uno cuadrado, centrado.
    
    Args:
        x, y: posición superior izquierda (en píxeles)
        w, h: ancho y alto (en píxeles)
        img_w, img_h: dimensiones de la imagen
    
    Returns:
        (sq_x, sq_y, sq_side): esquina superior izquierda y lado del cuadrado
    """
    side = max(w, h)

    new_x = x - (side - w) / 2
    new_y = y - (side - h) / 2

    new_x = max(0, new_x)
    new_y = max(0, new_y)

    if new_x + side > img_w:
        side = img_w - new_x
    if new_y + side > img_h:
        side = img_h - new_y

    return int(new_x), int(new_y), int(side)