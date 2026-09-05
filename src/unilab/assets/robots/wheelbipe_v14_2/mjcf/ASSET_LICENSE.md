# WheelBipe MJCF asset license

The WheelBipe V14 robot description and flat scene XML in this directory are
derived from the public
[`SCUTRobotLab/wheelbipe_ros2_sim2sim`](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim)
repository. The upstream project is released under the MIT License.

Copyright (c) 2025-2026 SCUTRobotLab (https://www.scutbot.cn/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The source snapshot used for this migration is commit
[`daa34f54d56cab91b3989d8152a7ce7b61092994`](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/tree/daa34f54d56cab91b3989d8152a7ce7b61092994).
UniLab's copy adds backend-facing compiler metadata, sensors, and explicit
control limits. The task start keyframe is intentionally kept in the separate
`locomotion_task.xml` fragment and is UniLab integration work.
