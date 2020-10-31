# Getting Started

## Train GAN and generate images
1. We will train GAN using the data in file gan_classifier/gan_data_tools/metadata.csv
    ```
    python gan_classifier/dcgan_covid.py --mode=train --bs=16 --epochs=150 --sample_interval=10
    ```
2. Now we will generate 544 images using the GAN
    ```
    python gan_classifier/dcgan_covid.py --mode=generate --image_c--image_count=544
    ```

## Prepare the combined dataset:
1. Prepare the combined dataset (Dataset-A) by merging 3 datasets
    ```
    python cnn_classifier/data_tools/prepare_data.py --use_generated=False
    ```
2. Prepare the second combined dataset (Dataset-B) by adding generated images to the combined dataset
    ```
    python cnn_classifier/data_tools/prepare_data.py --use_generated=True
    ```

## Train DCNN classifier models using the combined dataset1 (Dataset-A) and evaluate using test data split
1. Train DCNN network using Dataset-A in stage1 by freezing all layers except the final layer and select the best model (rename the best model to best_1_2_freeze.pth )
    ```
    python cnn_classifier/tools/trainer.py --mode train --freeze --checkpoint cnn_classifier/models/CovidAID_4_class.pth --bs 16 --save cnn_classifier/models 
    ```
2. Evaluate the best model (Model-A) in stage1
    ```
    python cnn_classifier/tools/trainer.py --mode test --checkpoint cnn_classifier/models/best_1_2_freeze.pth 
    ```
3. Train DCNN network end to end using Dataset-A in stage2 by initializing weights from the best model in stage1, and select the best model
    ```
    python cnn_classifier/tools/trainer.py --mode train --checkpoint cnn_classifier/models/best_1_2_freeze.pth --bs 8 --save cnn_classifier/models 
    ```
2. Evaluate the best model (Model-A) in stage2
    ```
    python cnn_classifier/tools/trainer.py --mode test --checkpoint cnn_classifier/models/best_1_2_no_freeze.pth 
    ```

## Train DCNN classifier models using the combined dataset2 (Dataset-B containing generated images) and evaluate using test data split
1. Train DCNN network using Dataset-B in stage1 by freezing all layers except the final layer and select the best model (rename the best model to best_1_1_freeze.pth )
    ```
    python cnn_classifier/tools/trainer.py --mode train --freeze --checkpoint cnn_classifier/models/CovidAID_4_class.pth --bs 16 --save cnn_classifier/models --equal_sampling=True
    ```
2. Evaluate the best model (Model-B) in stage1
    ```
    python cnn_classifier/tools/trainer.py --mode test --checkpoint cnn_classifier/models/best_1_1_freeze.pth 
    ```
3. Train DCNN network end to end using Dataset-B in stage2 by initializing weights from the best model in stage1, and select the best model
    ```
    python cnn_classifier/tools/trainer.py --mode train --checkpoint cnn_classifier/models/best_1_1_freeze.pth --bs 8 --save cnn_classifier/models --equal_sampling=True
    ```
2. Evaluate the best model (Model-B) in stage2
    ```
    python cnn_classifier/tools/trainer.py --mode test --checkpoint cnn_classifier/models/best_1_1_no_freeze.pth 
    ```

## Train AC-GAN model using combined dataset1 (Dataset-A) and evaluate using test data split
1. Train AC-GAN network using Dataset-A
    ```
    python gan_classifier/dcgan_ac_covid.py --mode=train --bs=16 --epochs=50 --sample_interval=10
    ```

2. Evaluate AC-GAN model  
    ```
    python gan_classifier/dcgan_ac_covid.py --mode=evaluate --bs=16 --checkpoint=gan_classifier/model_weights/dcgan_ac_covid/discriminator_weights.hdf5
    ```

## Evaluate CovidAID model using test data split
1. Train AC-GAN network using Dataset-A
    ```
    python cnn_classifier/tools/trainer.py --mode test --checkpoint cnn_classifier/models/CovidAID_4_class.pth
    ```
