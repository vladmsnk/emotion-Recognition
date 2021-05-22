from extract.extract import Extract
from build.buildm import Build
from configs.config import Config
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pickle
from keras.models import load_model




with open('../pickle/pickle1.p', 'rb') as handle:
    X = pickle.load(handle)

with open('../pickle/pickle2.p', 'rb') as handle:
    y = pickle.load(handle)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,\
                                                    random_state=42)

# buil = Build(X_train,y_train)
# model = buil.creat_conv_model()
# checkpoint = ModelCheckpoint(Config.model_path,monitor='val_acc',verbose=1,mode= 'max',\
#                             save_best_only = True, save_weights_only = False, period =1 )
# model.fit(X_train, y_train, epochs=Config.n_epoch, batch_size=Config.batch_size,shuffle= True, callbacks=[checkpoint])
# model.save(Config.model_path)


model = load_model('../model/mod.model')
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


