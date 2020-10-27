import csv
from datetime import datetime, timedelta, date
from forex_python.converter import get_rate

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

d=input("Introduceti data de inceput a istoricului (DD-MM-YYYY): ").split("-") #Preluarea datelor dureaza mult pentru perioade mari de timp (~7-8 sec pentru 30 de zile de curs valutar). Recomandat cel mult 1 an 
start_date=date(int(d[2]),int(d[1]),int(d[0]))
end_date=date.today()
c1=input("Introduceti codul ISO al monedei de referinta (ex:EUR,USD,RON): ")
c2=input("Introduceti codul ISO al monedei al carei echivalent vreti sa il aflati fata de moneda de referinta: ")
print("Asteptati pana se preiau datele")
with open(c1+c2+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for single_date in daterange(start_date, end_date):
            writer.writerow([get_rate(c1,c2,single_date),single_date.strftime("%d-%m-%Y")])
print("Datele au fost preluate si exportate in",c1+c2+'.csv')