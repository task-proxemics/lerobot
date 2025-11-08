## Troubleshooting


1. `No status packet error.`

    **Cause**: The motor ids are not being found potentially

    **Solution**: Unplug power and data cable on driver board.

2. `TimeoutError: Timed out waiting for frame from camera OpenCVCamera(0) after 200 ms. Read thread alive: True.`

    **Cause**: Don't ask. Don't Tell. 

    **Solution**: In the `wrist` and `up` camera configs, set the `fourcc` param to `MJPG`