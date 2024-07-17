# IVT-Filter-GazeData
 IVT filter to process raw gaze data. Based on the Tobii documentation
https://connect.tobii.com/s/article/Gaze-Filter-functions-and-effects?language=en_US

FixationFilter.py is the base fixation IVT Filter. 
2-D Normalized gaze data, timestamps, validity of gaze data, and 3-D gaze data (not necessarily required). Gaze data is in the format: 
Timestamp,LeftValidity,LeftGazePoint2DX,LeftGazePoint2DY,LeftGazePoint3DX,LeftGazePoint3DY,LeftGazePoint3DZ,LeftEyePosition3DX,LeftEyePosition3DY,LeftEyePosition3DZ, RightValidity,RightGazePoint2DX,RightGazePoint2DY,RightGazePoint3DX,RightGazePoint3DY,RightGazePoint3DZ,RightEyePosition3DX,RightEyePosition3DY,RightEyePosition3DZ, SystemTimestamp,ComputerTimestamp

The rest of the scripts are task-specific.
