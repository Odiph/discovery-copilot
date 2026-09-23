"""The floating overlay: the surface you actually look at during an interview.

What was called "the overlay" until now was a browser page, and a browser page
can be neither translucent nor always-on-top -- so it was never an overlay, it
was a second window to alt-tab to. This package is the real thing: a
borderless, 85%-opaque, always-on-top window that sits over the call.

**Tkinter, deliberately.** It is standard library, so the pinned-dependency
gate's literal 5 is untouched and there is no install step before an
interview. The cost is real and visible in the design: square corners, flat
fills, no blur and no shadow. That constraint is not fought here -- the
layout leans into big type and hard contrast, which is what a surface glanced
at for under a second during a live conversation wants anyway.

**It is a CLIENT, not the interview.** Tkinter's `mainloop()` and
`app.main.run_start`'s `asyncio.run()` both want the main thread, and
`app/web/server.py` documents why that loop must stay on it. Rather than
restructure the interview around the window, the window supervises
`python -m app start` as a CHILD PROCESS and renders the wire protocol that
process already broadcasts -- `hello`, `survey`, `suggestion`, `segment`.
Every message it draws is one the phone surface already receives, so the two
surfaces cannot drift, and the interview's threading model is untouched.

This does mean the overlay is a launcher, which the README's "one command,
one process, no second launcher" principle argues against. That principle was
written about a WEB PAGE spawning subprocesses, and it is stated plainly here
rather than quietly bent: a persistent always-present record button is a
launcher by nature, and that is what was asked for.
"""
