# O-Week Demo

## Run
### Environment setup
To run the code, please first make sure you have all required packages installed. If you use `conda`, then just run (replace `<env>` with a proper environment name).
```
conda create --name <env> --file requirements.txt
```

or 

*This would set the name of the new environment as `ml`*
```
conda env create -f ml.yml
```

After installed the environment, you can use `conda activate <env>` to activate the environment.

Then, remember to install `PyQt5`, you can use
```
conda install -c anaconda pyqt
```

You can also install all the required packages mannually by `pip`.

### Run the Application
To run the application with GUI, use
```
python oweek_demo.py
```

To test the original model, use
```
python ./image_animation.py -i ./Inputs/feynman.jpeg -c ./checkpoints/vox-cpk.pth.tar
```
`feynman.jpeg` can be changed into other images in the `Inputs` folder.

## Add more images
To add more source images, just put new images into `Inputs` directory. The application will automatically detect them.

## Credit
This application is based on [face_recognition](https://github.com/ageitgey/face_recognition) and
[Real_Time_Image_Animation](https://github.com/anandpawara/Real_Time_Image_Animation)
