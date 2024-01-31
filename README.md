# fpv-race-timer
Race timer tool that measures lap times from video recordings

## Getting Started

Follow these instructions to setup your machine with the necessary dependencies for using this tool.

### Prerequisites - TODO

### Installing - TODO


## How to Run

There are 2 parts to using this tool:
1. Annotating a test lap by to mark lap and segment locations
2. Processing DVR lap recordings to automatically extract timestamps using the annotations from (1)

**Annotating a Test Lap**
Pick video recording that contains a test run of the lap you want to annotate. The run through doesn't need to be perfect, but the more representative it is of your ideal lap the better. 
```
python3 scripts/gate-annotator.py -i /path/to/test-lap-video
```

Use the annotation tool to mark frames. When you are complete, hit save - the script will process the annotated frames and save a file (`gates.yaml`) containing some important metadata.

**Processing Subsequent Laps**

Process your DVR recordings of your laps using the `gates.yaml` file generated from the annotation step.

```
python3 scripts/lightglue-mvp.py -i /path/to/laps-video -g /path/to/gates.yaml
```

This will print out each frame that has been detected that best matches the annotated frames. You can take these frame ids, and multiply them by the frame rate to determine the time between each frame detection (i.e. lap and segment times).



## Authors

* **Vinay Rajur** - [vrajur](https://github.com/vrajur)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Lightglue
* Superpoint
