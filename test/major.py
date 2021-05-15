from extract.extract import Extract
from preprocess.prep import Preprocess
from build.buildm import Build
from configs.config import Config
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint



conf = Config()
extr = Extract(conf.audpath)
prep = Preprocess()
extr.__call__()
prep.__call__()
X,y = prep.build_rand_feat()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,\
                                          random_state=42)
buil = Build(X_train,y_train)
buil.__call__()
model = buil.creat_conv_model()
checkpoint = ModelCheckpoint(conf.model_path,monitor='val_acc',verbose=1,mode= 'max',\
                             save_best_only = True, save_weights_only = False, period =1 )
model.fit(X_train, y_train, epochs=conf.n_epoch, batch_size=conf.batch_size,\
          shuffle= True,validation_data=0.1, callbacks=[checkpoint])
model.save(conf.model_path)


