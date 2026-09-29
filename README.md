gc_counter.py

# a specific camera
python fgc_counter.py --camera 1

# a video file
python fgc_counter.py match.mp4

# a video stream URL
python fgc_counter.py rtsp://192.168.0.10/stream

After starting:

A window with the video opens. The SCORE and time are shown in the top-left corner.
If the red basket box is found, the basket frame appears immediately. If not, take a few shots and the program will find the basket (or press C and mark it with the mouse).
The console prints an address for your phone, e.g. http://192.168.1.5:8080. Open it on a phone connected to the same Wi-Fi network.
When you quit (Q / Esc), the program prints the final score and saves settings to goals.json.

Note: console messages are in Russian.

Keyboard controls
Key	Action
C	set the robot basket frame with the mouse (top-left corner → bottom-right corner)
H	mark a human basket (balls in it are not counted)
X	remove human basket zones
Z	remove robot basket frames
L	search for the basket again
↑ / ]	enlarge the last frame
↓ / [	shrink the last frame
+ / -	add / subtract a point manually
R	reset score and timer
N	switch to the next camera
Space	pause
Q / Esc	quit

On-screen colors:

yellow frame — robot basket (orange with ? — basket is not clearly visible right now); green flash — a ball was counted;
gray frame — human basket;
circles: gray — ball detected, purple — ball in flight, green — ball counted.
Command-line options
Option	Description
source	video path, URL, or camera index
--camera N	camera index
--list-cameras	list available cameras
--config FILE	settings file (default: goals.json next to the script)
--relearn	ignore the saved basket and find it again
--recolor	detect the ball color again
--roi x0,y0,x1,y1	set the basket by hand (in frame pixels)
--goals N	how many baskets to find (default 1)
--learn-shots N	how many shots are needed to find the basket
--width, --height, --fps	camera resolution and frame rate
--no-window	run without a window
--no-web	don't start the phone page
--save out.mp4	record video with overlays
--report DIR	save events.csv and summary.json to a folder
--set KEY=VALUE	change any setting (can be repeated)
Examples
bash
# count balls in a match recording, save a report and an annotated video
python fgc_counter.py match.mp4 --no-window --report results --save marked.mp4

# new field: find the basket and ball color from scratch
python fgc_counter.py --relearn --recolor

# set the basket manually
python fgc_counter.py --roi 400,200,600,420

# change settings
python fgc_counter.py --set learn_shots=2 --set web_port=9000
Settings (--set)

Main settings (see the Settings class in the code for the full list):

Setting	Default	Description
proc_max_side	960	processing frame size (smaller = faster)
auto_color	true	detect ball color automatically
hue_lo, hue_hi	5, 20	ball hue range (HSV) when auto_color=false
ball_radius	0	ball radius in pixels (0 = detect automatically)
learn_shots	3	shots needed to find the basket
n_goals	1	number of baskets to count
auto_red_box	true	find the red basket box by color
auto_human	true	automatically exclude the transparent human basket
stabilize	true	camera stabilization
moving_camera	false	mode for a camera that moves
lock_basket	true	keep the frame attached to the basket by its appearance
apriltag	true	use an AprilTag marker
tag_family	36h11	AprilTag family
tag_ids	""	tag ids to use (comma-separated)
undo_sec	0.8	seconds after a count during which a ball bouncing out cancels the point
web_port	8080	phone page port (0 = disabled)

Settings can also be stored in goals.json under "settings":

json
{
  "settings": {
    "learn_shots": 2,
    "web_port": 9000
  }
}
How it works
Motion + color. Each frame is compared with the previous one; moving blobs of the right color and size are treated as balls.
Stabilization. Camera shift is estimated with phase correlation and optical flow, so shaking isn't mistaken for ball motion.
Tracking. Balls are linked across frames into trajectories. A trajectory counts as a shot if the ball clearly rose upward.
Scoring. A point is awarded when a shot ball comes down and ends up inside the basket frame (or disappears inside it). If the ball flies out within undo_sec, the point is removed.
Basket learning. Landing points of shots are clustered; once learn_shots shots land in the same spot, that's the basket. The frame then snaps to the colored basket structure.
Report

With --report results you get:

results/events.csv — every event: time, frame, basket, +1/-1, score, and reason;
results/summary.json — final score and basket coordinates.
Troubleshooting

Black camera image:

check that the lens or privacy shutter isn't covered;
make sure no other app is using the camera (Zoom, Teams, browser, OBS) and close them;
try another camera: --camera 1 or press N;
Windows: Settings → Privacy → Camera — allow apps to access the camera.

Basket not found — press C and mark it by hand, or run with --relearn.

Ball not detected / poor counting — run with --recolor to detect the color again. For fast shots, a higher frame rate helps: --fps 60.

Phone can't open the page — the phone and computer must be on the same Wi-Fi network; check that the firewall isn't blocking port 8080.
