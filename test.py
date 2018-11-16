# This is the main testing script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
from keras.models import load_model
import mycoco

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('test')
    model = load_model(args.modelfile)
    ids = mycoco.query(args.categories, exclusive=False)
    if args.maxinstances:
        x = args.maxinstances
    else:
        x = len(min(ids, key=len))
    list1 = []
    for i in range(len(ids)):
        list1.append(ids[i][:x])
    print("Maximum number of instances are :" , str(x))
    imgiter = mycoco.iter_images(list1, [0,1], size=(200,200), batch=50)
    img_sample = next(imgiter)
    predictions = model.predict(img_sample[0])
    classes = [(1 if x >= 0.5 else 0) for x in predictions]
    correct = [x[0] == x[1] for x in zip(classes, img_sample[1])]
    z = sum(correct)
    print(z)
    print("Option A is implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB():
    mycoco.setmode('test')
    print("Option B not implemented!")

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Evaluate a model.")    
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
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
