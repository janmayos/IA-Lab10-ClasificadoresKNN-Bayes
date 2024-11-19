from Clasificador import leave_one_out,k_fold_cross_validation,hold_out_70_30,clasificador_1nn,clasificador_euclidiano
from datasets import crear_matriz_linea_linea_archivo,procesar_data,separar_caracteristicas_clases
from Excel import crear_excel

def ejecutar(rutas_data):
    dataexcelmatriz = []
    for ruta in rutas_data:
        matriz_file = crear_matriz_linea_linea_archivo(ruta)
        matriz_process = procesar_data(matriz_file)
        caracteristicas,clases = separar_caracteristicas_clases(matriz_process)

        dataexcel = []
        #clasificador euclidiano
        # Validación Hold Out 70/30
        dataexcel.append(ruta)
        acc_ho, cm_ho = hold_out_70_30(caracteristicas, clases, clasificador_euclidiano)
        dataexcel.append("{:.2f}%".format(acc_ho * 100))

        # Validación 10-Fold Cross-Validation
        acc_cv, cm_cv = k_fold_cross_validation(caracteristicas, clases, clasificador_euclidiano, k=10)
        dataexcel.append("{:.2f}%".format(acc_cv * 100))

        # Validación leave_one_out
        acc_lo, cm_lo = leave_one_out(caracteristicas, clases, clasificador_euclidiano)
        dataexcel.append("{:.2f}%".format(acc_lo * 100))
        
        #clasificador 1nn
        # Validación Hold Out 70/30
        acc_ho, cm_ho = hold_out_70_30(caracteristicas, clases, clasificador_1nn)
        dataexcel.append("{:.2f}%".format(acc_ho * 100))
        # Validación 10-Fold Cross-Validation
        acc_cv, cm_cv = k_fold_cross_validation(caracteristicas, clases, clasificador_1nn, k=10)
        dataexcel.append("{:.2f}%".format(acc_cv * 100))
        # Validación leave_one_out
        acc_lo, cm_lo = leave_one_out(caracteristicas, clases, clasificador_1nn)
        dataexcel.append("{:.2f}%".format(acc_lo * 100))
        dataexcelmatriz.append(dataexcel)
    print(dataexcelmatriz)
    crear_excel("Dataset accuracy",dataexcelmatriz,["Ruta dataset","Euclidiano HO","Euclidiano KF CV","Euclidiano LOO","1-NN HO","1-NN KF CV","1-NN LOO"])


if __name__ == "__main__":
    ejecutar(['iris/bezdekIris.data',"./Dataset/ClaseAB.data","./Dataset/ClaseXYZ.data"])