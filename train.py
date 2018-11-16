# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from keras import Model
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, Dense, Activation, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import mycoco

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('train')
    ids = mycoco.query(args.categories, exclusive=False)
    if args.maxinstances:
        x = args.maxinstances
    else:
        x = len(min(ids, key=len))
    list1 = []
    for i in range(len(ids)):
        list1.append(ids[i][:x])
    print("Maximum number of instances are :" , str(x))
    imgiter = mycoco.iter_images(list1, [0,1], batch=100)
    input_img = Input(shape=(200,200,3))
    # Encoder Layers
    x = Conv2D(8, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)

    # Decoder Layers
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    decode = Dense(1, activation="sigmoid")(x)
    
    model = Model(input_img, decode)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    filepath="/scratch/gusmohyo/checkfile.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(imgiter, steps_per_epoch=10, epochs=30, callbacks=callbacks_list, verbose=0)
    model.save(args.modelfile)
    print("Option A is implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB():
    mycoco.setmode('train')
    print("Option B not implemented!")

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")    
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('checkpointdir', type=str,
                        help="directory for storing checkpointed models and other metadata (recommended to create a directory under /scratch/)")
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))

    if len(args.categories) < 2:
        print("Too few categories (<2).")
        exit(0)

    print("The queried COCO categories are:")
    for c in args.categories:
        print("\t" + c)

    print("Executing option " + args.option)
    if args.option == 'A':
        optA()
    elif args.option == 'B':
        optB()
    else:
        print("Option does not exist.")
        exit(0)
