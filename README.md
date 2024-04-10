# Maritime Surveillance System

The project contains a thermal vision based maritime surveillance system. It includes the following options.
1. Object tracking: The algorithm is capable of tracking vessels, ships, jet skies, and humans
2. Vessel re-identification: re-identify vessels irrespective of their viewpoint using a database.
3. Activity detection: detect suspicious activities like human trafficking.

## Overall Architecture

<div align="center">
    <img src="/images/backbone.png" width="850" height="450" alt="overall architecure"/>
</div>

The framework proposed in this study comprises three primary subsystems: object tracking, vessel re-identification, and activity detection, as depicted in the above figure. The thermal video feed captured by the camera is directed towards the object tracking and activity detection subsystems. Subsequently, the tracking subsystem outputs identified objects, which are then forwarded to the re-identification subsystem. The outputs generated by all three subsystems are integrated into a user interface, facilitating the visualization of detected marine vessels, associated activities, and the corresponding re-identification results.

## 1. Object Tracking
Move into the object Object tracking folder and follow the instructions there.

A demo video of our object tracking algorithm is presented below.
<div align="center">
<a href="https://drive.google.com/file/d/1CH0uQ3gU0Lt2J2lBxkF8xpyFhlBoK3fG/view?usp=drive_link">
    <img src="/images/tracking_tn.png" width="640" height="450" alt="Video Name"/>
</a>
</div>
## 2. Vessel Re-identification
Move into the object Object tracking folder and follow the instructions there.

The visual performance of our algorithm is presented below.

<div align="center">
    <img src="/images/reid_visual_res.PNG" width="850" height="450" alt="re-id"/>
</div>

Visual results of the re-identification algorithm. Each row includes two images depicting the same vessel and one image of a distinct vessel with minor alterations. Our algorithm accurately distinguishes those images in column 2 that pertain to the vessel category in column 1 rather than the vessel category in column 3. Note that the algorithm demonstrates proficiency despite challenges such as orientation variations and blurred images.
