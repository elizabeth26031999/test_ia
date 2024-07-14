import os
import tensorflow as tf
from object_detection import model_lib_v2

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('pipeline_config_path', '', 'Path to pipeline config file')
flags.DEFINE_string('model_dir', '', 'Path to model directory')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps')
flags.DEFINE_bool('sample_1_of_n_eval_examples', 1, 'Number of eval steps to sample')
flags.DEFINE_bool('alsologtostderr', False, 'Also log to stderr')

def main(unused_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    model_lib_v2.train_loop(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        use_tpu=False,
        checkpoint_every_n=1000,
        record_summaries=True)

if __name__ == '__main__':
    tf.compat.v1.app.run()



# import tensorflow as tf
# from object_detection import model_lib_v2

# # Ruta al directorio del modelo preentrenado
# model_dir = 'model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

# # Configuración de la ruta al pipeline.config
# pipeline_config_path = os.path.join(model_dir, 'pipeline.config')

# # Parámetros de entrenamiento adicionales si es necesario
# train_params = {
#     'model_dir': model_dir,
#     'pipeline_config_path': pipeline_config_path,
#     # Otros parámetros como num_train_steps, num_eval_steps, etc.
# }

# # Configuración de GPU si es necesario
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for physical_device in physical_devices:
#     tf.config.experimental.set_memory_growth(physical_device, True)

# # Iniciar el entrenamiento
# model_lib_v2.train_loop(**train_params)
