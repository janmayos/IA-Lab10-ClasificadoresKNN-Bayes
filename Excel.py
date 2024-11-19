from openpyxl import Workbook

# Crear un nuevo libro de Excel
def crear_excel(nombre_archivo,matriz_info,cabeceras):

    wb = Workbook()
    ws = wb.active  # Seleccionar la hoja activa

    # Añadir datos a la hoja
    ws.append(cabeceras)
    for line in matriz_info:
        ws.append(line)
    
    # Guardar el archivo
    wb.save(nombre_archivo+".xlsx")