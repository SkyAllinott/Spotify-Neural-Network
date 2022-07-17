# Flags
FLAGS <- flags(flag_integer('dense_units1', 15),
               flag_numeric('dropout1', 0.1),
               flag_integer('batch_size', 32),
               flag_integer('epochs', 50))

# Model
tensorflow::set_random_seed(1234)
model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$dense_units1, activation = 'relu', input_shape = c(14)) %>%
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(units = 1)

# Compile
model %>% compile(loss = 'mse', 
                  optimizer = 'rmsprop',
                  metrics = 'mae')

# Fit
early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 10)
history <- model %>%
  fit(training,
      trainingtarget,
      epochs=100,
      batch_size=FLAGS$batch_size,
      validation_split=0.2,
      verbose = 0,
      callbacks = list(early_stop))