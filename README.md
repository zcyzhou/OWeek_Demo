# O-Week Demo

## Run

To Run the demo with a rough GUI, use
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
