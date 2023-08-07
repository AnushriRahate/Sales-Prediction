import pandas as pd
from joblib import load
from tkinter import *
from tkinter import messagebox

sc=load("scaling.joblib")
lc=load("reg.joblib")
lf=load("fat.joblib")
lt=load("type.joblub")
ll=load("out.joblib")
lg=load("outtype.joblib")

def result():
        new=pd.DataFrame({"Item_Weight":[float(a2.get())],"Item_Fat_Content":[(a3.get())],"Item_Type":[(a4.get())],"Item_MRP":[float(a5.get())],"Outlet_Location_Type":[(a6.get())],"Outlet_Type":[(a7.get())]})
        new['Item_Fat_Content']=lf.transform(new['Item_Fat_Content'])
        new['Item_Type']=lt.transform(new["Item_Type"])
        new['Outlet_Location_Type']=ll.transform(new['Outlet_Location_Type'])
        new['Outlet_Type']=lg.transform(new['Outlet_Type'])
        new=sc.transform(new)
        res=lc.predict(new)
        ans=Label(root,text=res,font=('Arial',10)).place(x=200,y=260)



root=Tk()
root.geometry("450x450")
root.resizable(0,0)
root.title("Real Estate Prediction")


a2=StringVar()
a3=StringVar()
a4=StringVar()
a5=StringVar()
a6=StringVar()
a7=StringVar()

heading=Label(root,text="Sales Prediction System",font=("Arial",20),fg="PURPLE")
heading.place(x=10,y=5)


three=Label(root,text="Item Weight")
three.place(x=10,y=80)
four=Entry(root,textvariable=a2)
four.place(x=200,y=80)

five=Label(root,text="Item Fat Content")
five.place(x=10,y=110)
six=Entry(root,textvariable=a3)
six.place(x=200,y=110)

seven=Label(root,text="Item Type")
seven.place(x=10,y=140)
eight=Entry(root,textvariable=a4)
eight.place(x=200,y=140)

nine=Label(root,text="Item MRP")
nine.place(x=10,y=170)
ten=Entry(root,textvariable=a5)
ten.place(x=200,y=170)

nine=Label(root,text="Outlet Location Type")
nine.place(x=10,y=200)
ten=Entry(root,textvariable=a6)
ten.place(x=200,y=200)

eleven=Label(root,text="Outlet Type")
eleven.place(x=10,y=230)
one1=Entry(root,textvariable=a7)
one1.place(x=200,y=230)

submit=Button(root,text="PREDICT",bg="lightgrey",command=result,font=("Arial"),fg='black')
submit.place(x=10,y=260)

root.mainloop()