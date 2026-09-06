# Walking Assistant

Camera-based walking guidance for blind and low-vision users, built on RT-DETRv2.

It watches the scene through a camera, works out how far away things are and where
the floor is, and tells you what matters through sound: short spoken phrases for
what is around you, and a beep that speeds up as you get closer to something.

**This is not a safety device.** It is experimental software running on a camera
that can be wrong, in the dark, in the rain, or pointed at the sky. It does not
replace a white cane, a guide dog, or your own judgement. Use it as extra
information on top of the mobility aids you already trust, never instead of them.

This page explains what the sounds mean. For code and setup, see
`RT-DETR/rtdetrv2_pytorch/references/deploy/readme.md`.

---

## What you hear

There are two separate sound channels, and they are doing different jobs.

The **beeps** are the fast channel. They run continuously, they never wait for
anything, and they tell you *how close* and *which side*. The **speech** is the
slow channel. It tells you *what* the thing is. Speech is deliberately rare,
because a spoken sentence takes over a second to say, and in that time you have
already taken two steps.

When nothing is near and the way ahead is open, **you hear nothing at all.**
Silence means the path is clear. That is the normal state, and it is deliberate:
a system that talks constantly is one you stop listening to.

### The beeps

Beeping starts when something comes within **4 metres**. Nothing beeps beyond that.

**Speed means distance.** At 4 metres you get about one beep per second. As you
close in, the beeps come faster and faster, until at half a metre they are almost
a continuous tone. You do not have to count anything — the acceleration is the
signal, the way a parking sensor works.

**Left and right ears mean left and right.** The beep is panned toward whatever
it is warning you about. Something directly ahead sounds centred in both ears.
Something off to your left sounds in your left ear, and the further to the side
it is, the harder it is panned. **Use stereo headphones or earbuds**, or this
information is lost entirely.

**Pitch means what kind of hazard.** There are only two, and they are far apart
on purpose so you can never confuse them:

- A **mid-pitched steady beep** (around 660 Hz, roughly the E above middle C) is
  an ordinary obstacle. A person, a wall, a bin, a parked car.
- A **low two-tone chirp** (around 220 Hz, an octave and a half lower) is a
  **drop-off** — the ground falling away. A kerb, a step down, the edge of a
  platform. If you hear the low sound, slow down.

Keep one ear free, or use bone-conduction headphones. You still need to hear
traffic and your cane.

### The speech

At most one short phrase every **two seconds**, and usually much less than that.

**Objects** are announced as *what, where, how far*, always in that order, so you
can understand the first word and stop listening if you need to:

> "person, slightly left, 2.5 metres"
> "chair, ahead, 2 metres"
> "bicycle, right, 3 metres"

The direction words are **ahead** (roughly straight in front), **slightly left**
or **slightly right** (a little off to one side), and **left** or **right** (well
off to the side). Distances are spoken to the nearest half metre up to 3 metres,
and to the nearest metre beyond that. Anything under a metre is just **"very
close"**, because at that range a number is more precision than the estimate
deserves.

**When something is very close or closing fast, the distance is dropped:**

> "person, ahead"

That is not the system being vague. Under about 1.5 metres, or when something is
going to reach you within 1.5 seconds, the number would be out of date before the
sentence finished. Getting the warning to you sooner matters more.

**Drop-offs** get their own wording, and it depends on whether it is in your way:

> "stop, step down"  — the ground drops away directly in your path. Stop walking.
> "step down, left"  — the ground drops away beside you. Usually the edge of the
> path or pavement you are on. Worth knowing, not a reason to stop.

**Steering** happens when the way ahead is blocked but there is a clear way past:

> "bear left"
> "bear right"

It only says this when there is somewhere better to go. If everything ahead is
blocked, it will not send you sideways into something else.

**Repetition is limited on purpose.** Once something has been announced, it is not
mentioned again for 8 seconds — unless it gets significantly closer, which counts
as new information. So walking past a bench mentions it once, not thirty times.

### Quick reference

| What you hear | What it means |
|---|---|
| Nothing | Path ahead is clear. Keep going. |
| Slow mid beep | Something about 4 metres away |
| Fast mid beep | Something about 1 metre away |
| Almost continuous mid tone | Something within half a metre |
| Low two-tone chirp | **Drop-off — ground falling away** |
| Beep only in one ear | Hazard is on that side |
| "person, ahead, 2 metres" | What it is, where, how far |
| "person, ahead" | Close or closing fast; act now |
| "stop, step down" | **Step down in your path. Stop.** |
| "step down, left" | Path edge on your left |
| "bear left" / "bear right" | Blocked ahead; clearer this way |

---

## What a real walk sounds like

You set off down a corridor. **Silence.** Nothing within 4 metres, floor found,
way ahead open.

Someone walks toward you. At 3 metres a slow beep starts, centred, and you hear
*"person, ahead, 3 metres"*. The beeping quickens as they approach. They drift to
your left; the beep drifts into your left ear with them.

They stop in the middle of the corridor. The beep is now fast. You hear *"bear
right"* — there is room to pass on that side. You move right, the beeping slows
as the gap opens, and once you are past, **silence** returns.

You reach the door and step outside onto the pavement. Nothing to announce, so
nothing is said. You walk on.

You approach a kerb. The beep drops to that **low two-tone chirp** and you hear
*"stop, step down"*. You stop, find the kerb edge with your cane, and step down
in your own time.

Now on the pavement beside a road. Every so often you hear *"step down, left"* —
that is the kerb edge running along beside you. It is not telling you to stop; it
is telling you where the edge of your pavement is.

A car pulls out ahead. Because it is *moving toward you*, it is treated as urgent
even though it is still 5 metres away: *"car, ahead"*, no distance, with a fast
beep. Something closing at speed is a different matter from something parked in
the same spot.

---

## What it can and cannot see

**It finds by recognition:** people, bicycles, cars, motorbikes, buses, lorries,
dogs, cats, horses, benches, chairs, sofas, beds, tables, toilets, potted plants,
fridges, televisions, suitcases, backpacks, traffic lights, stop signs, fire
hydrants and parking meters.

**It finds by shape, without needing to recognise anything:** anything standing up
off the floor in your way, and any ground that drops away — kerbs, steps down,
platform edges, holes. This is important, because there is no such thing as a
"kerb detector"; drop-offs are found purely from the shape of the ground, which is
also why they work for obstacles nobody trained a model on.

**It cannot see:** doors and doorways, stairs going *up* as distinct from a wall,
poles, posts, overhanging branches or open cabinet doors at head height, glass,
kerbs it cannot see the ground beyond, potholes smaller than about 15 cm deep,
anything behind you, and anything outside the camera's roughly 65-degree cone of
view.

**It works badly:** in the dark, in heavy rain, pointed at a blank wall or the
sky, or when the camera cannot see the ground in front of you.

**It does not know the difference between a footpath and the grass beside it.**
It reasons about shape, not surface. If a person blocks the path and the verge is
flat, it may well suggest stepping onto the verge, because to it that is simply
open ground.

---

## How to wear and hold it

Hold the phone or camera at **chest height**, upright, pointing straight ahead and
very slightly down, so it can see the ground about 2 to 5 metres in front of you.

Two things depend on this. Distances are computed from the shape of the ground, so
a camera pointed at the ceiling or at your feet has nothing useful to work with.
And the system checks that the floor it finds is between **0.8 and 1.8 metres**
below the camera — a plausible height for a person carrying one. Outside that
range it decides it has not found the floor, and the path guidance switches off
while object announcements carry on.

A chest harness or lanyard works better than holding it in your hand, because your
hand swings as you walk and the horizon swings with it.

---

## Running it

Full setup is in `RT-DETR/rtdetrv2_pytorch/references/deploy/readme.md`. In short:

```
cd C:\obj-detection\RT-DETR\rtdetrv2_pytorch

python references/deploy/rtdetrv2_assist.py ^
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml ^
  -r weights/rtdetrv2_r50vd_6x_coco_ema.pth ^
  --source 0 --device cuda:0 --overlay
```

`--source 0` is the laptop webcam. `--overlay` shows a debug window for a sighted
helper. `--no-audio` runs it silently and prints every phrase it would have said,
which is the quickest way to judge whether it is talking too much.

### Using your phone's camera

There is no phone app yet. What works today is streaming the phone's camera to the
computer over WiFi, which is a genuinely useful way to test with real optics while
walking around:

1. Install an IP camera app on the phone (on Android, *IP Webcam* is the usual
   choice; on iOS, *EpocCam* or *iVCam*).
2. Start its server. It will show an address like `http://192.168.1.42:8080`.
3. Check that address opens in a browser on the computer first.
4. Run with `--source http://192.168.1.42:8080/video` and `--hfov 67`.

Audio comes out of the **computer**, not the phone, so wear headphones plugged
into the computer. Expect an extra fraction of a second of lag over WiFi; that is
the network, not the software.

### The one setting you must get right

`--hfov` is the camera's horizontal field of view in degrees. **Every distance
scales with it.** The default of 65 suits most laptop webcams and phone main
cameras. An ultrawide phone lens is more like 110 and will read badly wrong at 65.

To check it: stand a measured distance from the camera and see whether the spoken
distance matches. If everything sounds too close, your real field of view is
narrower than the setting; raise it if things sound too far. Get this right before
judging anything else.

---

## How it works, briefly

An object detector (RT-DETRv2) finds and names things in each frame. Separately, a
depth network estimates the shape of the scene — but only *relatively*, on an
unknown scale, so on its own it cannot say "two metres".

The scale comes from the detector. A person is about 1.7 metres tall, a car about
1.5. Knowing how tall something really is and how tall it appears gives its true
distance, and enough of those pin the whole depth map to real metres. **The
detector calibrates the depth network**, which is how one ordinary camera produces
real distances with no depth sensor, no second lens, and nothing for the user to
set up.

With real distances, the scene becomes 3D points. The floor is found afresh every
frame, and everything is measured against it: standing up off the floor means
obstacle, sitting below it means drop-off. The walkable area is then swept for a
clear route, widened by half a shoulder width first so that "a clear line" means
"a person actually fits".

When the depth scale cannot be worked out — nothing of known height in view, such
as an empty corridor — the system says directions without distances rather than
stating a number it does not trust.

---

## Project status

Working today: the guidance system itself, running on a Windows laptop with an
NVIDIA GPU, from a webcam, a video file, or a phone streaming over WiFi. About 15
frames per second on an RTX 4060. 131 automated tests.

Not built yet: a standalone Android app. The guidance logic was written to be
independent of any camera or speaker precisely so it can be moved onto a phone
later, but that work has not started. The next step is exporting the models and
measuring whether they run fast enough on a phone at all.

Known limits are listed under "What it can and cannot see" above. The most
significant are that it cannot tell a path from the ground beside it, that it
misses head-height obstacles entirely, and that distances depend on the field of
view being set correctly.
