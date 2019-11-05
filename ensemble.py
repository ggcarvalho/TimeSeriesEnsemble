
################################### IMPORTING #######################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
from mlp_models import *
sns.set()
#####################################################################################################

data = pd.read_csv("BTC-USDw.csv")
data = data["Close"]
data =  np.log(data)
serie = data

plt.figure(1)
plt.xlabel("Weeks")
plt.ylabel("Log(price)")
plt.title("BTC - USD")
plt.plot(serie)
plt.savefig("btc-usd.png")
plt.close()


plt.figure(2)
treino = serie.loc[:284]
val = serie.loc[285:379]
teste = serie.loc[380:474]
plt.plot(treino,label="Train",color="darkblue")
plt.plot(val,label="Val",color="green")
plt.plot(teste,label="Test",color='coral')
plt.legend(loc='best',fontsize="x-small")
plt.xlabel("Weeks")
plt.ylabel("Log(price)")
plt.title("BTC - USD")
plt.savefig("split.png")
plt.close()




def gerar_janelas(tam_janela, serie):
    # serie: vetor do tipo numpy ou lista
    tam_serie = len(serie)
    tam_janela = tam_janela +1 # Adicionado mais um ponto para retornar o target na janela

    janela = list(serie[0:0+tam_janela]) #primeira janela p criar o objeto np
    janelas_np = np.array(np.transpose(janela))

    for i in range(1, tam_serie-tam_janela):
        janela = list(serie[i:i+tam_janela])
        j_np = np.array(np.transpose(janela))

        janelas_np = np.vstack((janelas_np, j_np))


    return janelas_np


def select_lag_acf(serie, max_lag):
    from statsmodels.tsa.stattools import acf
    x = serie[0: max_lag+1]

    acf_x, confint = acf(serie, nlags=max_lag, alpha=.05, fft=False,
                             unbiased=False)

    limiar_superior = confint[:, 1] - acf_x
    limiar_inferior = confint[:, 0] - acf_x

    lags_selecionados = []

    for i in range(1, max_lag+1):


        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i-1)  #-1 por conta que o lag 1 em python é o 0

    #caso nenhum lag seja selecionado, essa atividade de seleção
    # para o gridsearch encontrar a melhor combinação de lags
    if len(lags_selecionados)==0:


        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    print('LAGS', lags_selecionados)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #inverte o valor dos lags para usar na lista de dados se os dados forem de ordem [t t+1 t+2 t+3]
    lags_selecionados = [max_lag - (i+1) for i in lags_selecionados]
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    return lags_selecionados

def split_serie_with_lags(serie, perc_train, perc_val = 0):

    #faz corte na serie com as janelas já formadas

    x_date = serie[:, 0:-1]
    y_date = serie[:, -1]

    train_size = np.fix(len(serie) *perc_train)
    train_size = train_size.astype(int)

    if perc_val > 0:
        val_size = np.fix(len(serie) *perc_val).astype(int)


        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]
        print("Particao de Treinamento:", 0, train_size  )

        x_val = x_date[train_size:train_size+val_size,:]
        y_val = y_date[train_size:train_size+val_size]

        print("Particao de Validacao:",train_size, train_size+val_size)

        x_test = x_date[(train_size+val_size):-1,:]
        y_test = y_date[(train_size+val_size):-1]

        print("Particao de Teste:", train_size+val_size, len(y_date))

        return x_train, y_train, x_test, y_test, x_val, y_val

    else:

        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]

        x_test = x_date[train_size:-1,:]
        y_test = y_date[train_size:-1]

        return x_train, y_train, x_test, y_test
tam_janela = 4
serie_janelas = gerar_janelas(tam_janela, serie)
x_train, y_train, x_test, y_test, x_val, y_val = split_serie_with_lags(serie_janelas, 0.6,
 perc_val = 0.2)

models = gen_all_models(20)
lags = len(x_train[0])

mse_list = []
for model in models:
    model.fit(x_train[:,-lags:], y_train)
    predict_validation = model.predict(x_val[:,-lags:])
    mse = MSE(y_val, predict_validation)
    mse_list.append(mse)

mse_list = np.array(mse_list)
best_mse_ind = mse_list.argsort()[:10]


best_10 = n_best_models(models,best_mse_ind)

predictions = []
for model in best_10:
    predict_test = model.predict(x_test[:, -lags:])
    predictions.append(predict_test)

mean_pred = np.mean(predictions, axis=0)
median_pred = np.median(predictions, axis=0)

plt.figure(3)
for pred in predictions:
    plt.plot(pred,linewidth=0.7)
plt.plot(mean_pred,label="Mean")
plt.plot(median_pred,label="Median")
plt.legend()
plt.show()
plt.close()


plt.figure(4)
plt.plot(median_pred,label="Median",linewidth=1)
plt.plot(mean_pred, label="Mean",linewidth=1)
plt.plot(y_test,label = "Test",linewidth=1)
plt.legend()
plt.show()
plt.close()

print("MSE Test (Mean) = %s" %MSE(y_test, mean_pred))
print("MSE Test (Median) = %s" %MSE(y_test, median_pred))
