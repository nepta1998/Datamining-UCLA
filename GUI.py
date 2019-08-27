from tkinter import *
import usarModelo as um


def prdiccionModel():
    w=um.prediccion(entry.get())
    z=w[0]
    if(z==0):
        prediccion.set("0.Incorrecto")
    else:
        if(z == 1):
           prediccion.set("1.Correcto")
        else:
            if(z == 3):
                prediccion.set("3.Dudoso")

    


ventana = Tk()
ventana.geometry('400x460+100+100')
ventana.configure(background='dark turquoise')
ventana.title("Mineria de datos")
titulo = Label(ventana, text='Mineria de datos',font=("Forte", 26, "bold"), bg='dark turquoise')
titulo.pack()
imagen = PhotoImage(file="hola.png")
Label(ventana, image=imagen, bg="dark turquoise").pack(pady=5)
Label(ventana, text='Palabra/Oracion:', font=("Futura Md BT", 18), bg='dark turquoise').pack(fill=X)
entry = Entry(ventana)
entry.pack(fill=X, padx=5, pady=5, ipadx=2, ipady=5)

Label(ventana, text='Prediccion:', font=("Futura Md BT", 18), bg='dark turquoise').pack(fill=X)
prediccion = StringVar()
entry1 = Entry(ventana, state="readonly", textvariable=prediccion)
entry1.pack(fill=X, padx=5, pady=5, ipadx=2, ipady=5)
boton1 = Button(ventana, text="Predecir", command=prdiccionModel,font=("Agency FB", 20), bg="medium turquoise")
#boton2 = Button(ventana, text="Re-Entrnar Modelo",font=("Agency FB", 20), bg="medium turquoise")
boton1.pack(fill=X, padx=5, pady=5, ipadx=2, ipady=5)
#boton2.pack(fill=X, padx=5, pady=5, ipadx=2, ipady=5)
#lblprediccion.text("hola")

ventana.mainloop()
