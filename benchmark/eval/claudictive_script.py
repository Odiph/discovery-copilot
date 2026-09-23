"""A synthetic 25-minute discovery interview for the Claudictive demo project.

**Entirely invented.** Nothing here was said by a real person, and
`benchmark/data/real/` was not consulted. The domain is real (an in-game Claude
Code overlay shipped through Overwolf) because the demo project is built on it.

Written to exercise the demo's five hypotheses without simply confirming them:
the interviewee contradicts h1 in one place, is ambivalent about h4, and gives
the interviewer several chances to ask a leading question. An interview where
everything lines up would make a poor demo of a tool whose whole thesis is
disconfirmation.

Survey questions from `projects/demo-claudictive/context/survey.json` are asked
close to verbatim so the live coverage detector has something real to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.console import use_utf8_console  # noqa: E402

# (speaker, text). "interviewer" and "interviewee" get different SAPI voices.
TURNS: "list[tuple[str, str]]" = [
    ("interviewer", "Thanks for making the time. Before we get into any of it, I mostly want to hear how you actually work, not what you think of the product. So there are no wrong answers here."),
    ("interviewee", "Sure. That's fine. I've done a few of these before, so I know the drill."),
    ("interviewer", "Walk me through the last time you sat down to play, from turning the machine on to the game actually starting."),
    ("interviewee", "Okay, so, that would have been Thursday night. The machine is never off, it just sleeps, so I wake it up and there's usually about ten browser tabs still open from work and a terminal that's been sitting there since the afternoon."),
    ("interviewee", "I close some of the tabs, not all of them, and then I open the launcher. And then I wait, because the launcher is always updating something. Every single time. I don't think I have ever opened it and had it just be ready."),
    ("interviewer", "How long is that wait, usually?"),
    ("interviewee", "A minute? Two if it's a bad night. It's not long enough to go and do something else, which is the annoying part. It's exactly long enough to be dead time."),
    ("interviewer", "What do you do during it?"),
    ("interviewee", "Phone, mostly. Which I'm not proud of. Sometimes I'll go back to the terminal and look at whatever I was doing earlier, but I try not to, because if I start reading code I'll still be reading it forty minutes later and then I haven't played."),
    ("interviewer", "Say more about that. What happens when you do start reading?"),
    ("interviewee", "It's just, the context comes back. If I look at the thing I was stuck on in the afternoon, I remember why I was stuck, and then I want to try the one idea I had in the shower, and then it's eleven o'clock and my friends have gone offline."),
    # --- follow-up bait: a vague quantifier and an unexplored strong word ---
    # A generalisation with no instance behind it, and a loaded word dropped
    # and walked past. Both are exactly what a deepening suggestion is for,
    # and the interviewer moves on without taking either.
    ("interviewee", "It's always the same, really. Every night is basically that, and it's been like that for years now. It's a bit grim when you say it out loud."),
    ("interviewer", "Right. So you're protecting the play time from the work."),
    ("interviewee", "Yeah, I suppose I am. I hadn't put it like that. It's more that I know myself. If I open that door I'm going through it."),
    ("interviewer", "Tell me about the last time you needed something from outside the game while you were playing."),
    ("interviewee", "That happens constantly. Not always work things. Sometimes it's, what's this item worth, or what's the actual counter to this character, and that's a wiki thing."),
    ("interviewee", "But the work ones happen too. I'll be mid-match and something will surface. Like, oh, that's why the migration failed. And I've got nowhere to put it."),
    ("interviewer", "What do you do with it when it surfaces?"),
    ("interviewee", "Honestly? Mostly nothing. I tell myself I'll remember it. I have a note app on my phone that I use maybe one time in five, and when I do use it the note says something like, quote, the thing with the dates, unquote, which is no help at all the next morning."),
    ("interviewer", "Tell me about a specific one. The most recent one you can remember."),
    ("interviewee", "That's the joke, isn't it. I can't. I know there were about three of them last Thursday. I remember the feeling of having them. I could not tell you what a single one of them was."),
    ("interviewer", "Three in one evening."),
    ("interviewee", "Roughly. It's usually two or three on a work night. Fewer on a weekend, because I'm not carrying anything in."),
    # SURVEY QUESTION s3 IS DELIBERATELY NEVER ASKED. The interviewer skips
    # past it to save time, which is the ordinary way coverage is lost -- not
    # forgetting, but triaging under a clock. The subject then volunteers the
    # answer anyway, which is the harder case for a coach: the TOPIC was
    # covered while the QUESTION was not, and only one of those is recorded.
    ("interviewer", "Let me skip ahead, I want to be sure we get to the install before we run out of time."),
    ("interviewee", "Sure. Although, one thing on the setup, because I think it matters. I've got one screen. It's a big ultrawide, but it's one screen, and the desk physically will not take another one."),
    ("interviewer", "Was that a decision or just how it worked out?"),
    ("interviewee", "Bit of both. I had two smaller ones before and I got sick of the bezel down the middle. So I sold both and bought the wide one. And then about a month later I realised I'd given up the thing where you can have the game on one and everything else on the other."),
    ("interviewer", "Do you regret it?"),
    ("interviewee", "Not for work. For work the wide one is much better, I can have three files side by side. For gaming it's worse, and I knew that when I did it, I just weighted it wrong."),
    ("interviewer", "So when you're playing, the whole screen is the game."),
    ("interviewee", "The whole screen is the game. Discord is on my phone, propped up next to the keyboard, because it has to live somewhere."),
    ("interviewer", "That's really useful. Tell me about the last time you alt-tabbed out mid-session. What happened when you came back?"),
    ("interviewee", "Right, so, this depends on the game. In the shooter I basically don't, because it's five people relying on me and thirty seconds is a lost round. In the strategy one I do it all the time, because it's turn-based and nobody's waiting."),
    ("interviewer", "Take me through the last time in the shooter."),
    ("interviewee", "That was a while ago and it went badly. I tabbed out to check something, I was maybe fifteen seconds, and I came back and I was dead on the floor and someone was typing at me in the chat."),
    ("interviewer", "And after that?"),
    ("interviewee", "After that I stopped. I genuinely have not done it in that game since. It's not even the dying, it's that four other people had a worse time because I wanted to look something up."),
    ("interviewer", "And in the strategy game?"),
    ("interviewee", "Constantly. It's borderless windowed, I've got it set up that way on purpose, and I'm in and out of it the whole evening. Nobody notices and nothing is lost."),
    ("interviewer", "What is different about coming back, between those two?"),
    ("interviewee", "In the strategy game, coming back is fine, the board is still the board. The cost isn't the leaving, it's remembering what I was doing before I left. That's the bit that gets me."),
    ("interviewer", "Say more about that?"),
    ("interviewee", "I'll tab out to check one thing, and while I'm out there I'll see a message, and then I'm answering that, and when I come back to the game I've lost the plan I had. I know what the board says. I don't know what I was going to do about it."),
    ("interviewer", "Walk me through installing a new tool on your gaming machine the last time you did it."),
    ("interviewee", "The last one was actually your thing. That was, what, two weeks ago?"),
    ("interviewer", "Take me through it exactly as it happened."),
    ("interviewee", "So I found the site, I read maybe half of the front page, and I clicked the download. And then what I got was not the thing. It was an installer for something else called Overwolf, which I had heard of but did not have."),
    ("interviewer", "What went through your head at that point?"),
    ("interviewee", "I stopped. I definitely stopped. It's the classic thing where you wanted one program and now you're being asked to install a platform, and my first thought was, what is this, and my second thought was, what does it get to see."),
    ("interviewer", "What did you do next?"),
    ("interviewee", "I went and looked it up. Fifteen, twenty minutes probably. Found out it's what a lot of the game overlays run on, found out roughly what the permissions model is, decided it was fine and went back and installed it."),
    ("interviewer", "So you did continue."),
    ("interviewee", "I did continue. But I want to be clear that I'm not the average person there. I install things for a living. I know how to go and check. Half the people I play with would have closed the tab at that screen and never thought about it again."),
    ("interviewer", "What makes you say that?"),
    ("interviewee", "Because I've watched them do it with other things. Anything that asks for a second install, or wants an account before it shows you anything, they're gone. The bar is really low for closing a tab."),
    ("interviewer", "Think of a problem you hit mid-game that you could not solve there and then. What did you actually do next?"),
    ("interviewee", "Okay. So a couple of weeks ago I had a build failing on the machine, and I'd started it and then queued up, because that's what I do, the build takes eight minutes and a queue takes four."),
    ("interviewer", "You start the build and then queue."),
    ("interviewee", "Always. If I'm going to be waiting anyway I'd rather be waiting for two things at once. And honestly if the build finishes first I'm slightly annoyed at the queue, which tells you something about how my brain works."),
    ("interviewer", "What happened with the failing build?"),
    ("interviewee", "It failed while I was in the match. I could see the notification but I couldn't read it properly, and I couldn't do anything about it, so I sat there for the rest of the round half-thinking about it and playing badly."),
    ("interviewer", "And then?"),
    ("interviewee", "And then the round ended and I read it, and it was something stupid, a version pin. Took two minutes. But I'd spent twenty minutes carrying it around."),
    ("interviewer", "What would have made that better?"),
    ("interviewee", "If I could have just seen the error. Not fixed it, just seen it. Because then I'd have known it was a two minute thing and I could have stopped thinking about it."),
    ("interviewer", "Just seeing it would have been enough?"),
    ("interviewee", "For that one, yes. Because the anxiety wasn't the bug, it was not knowing how bad the bug was."),
    ("interviewer", "Is that generally true, or was that one unusual?"),
    ("interviewee", "Hmm. No, I think if I'm honest, seeing it is usually not enough. Seeing it tells me whether to worry. But if I can see it and I know it's a two minute fix, then I want to do the two minute fix, and now I'm asking for a lot more than a window that shows me things."),
    ("interviewer", "Tell me more about that."),
    ("interviewee", "It's like, if you show me the error and I can't act on it, you've moved the annoyance, you haven't removed it. Now instead of not knowing, I know and I'm stuck. I'm not sure that's better. It might be worse."),
    ("interviewer", "That's a really important distinction. Can you give me an example where knowing and being stuck was worse?"),
    ("interviewee", "The dependency thing, actually. Once I knew what it was, I spent the next two rounds mentally writing the fix instead of playing. If I hadn't known I'd have just played and dealt with it after."),
    ("interviewer", "So there's a version of this that makes your evening worse."),
    ("interviewee", "There's definitely a version of this that makes my evening worse. I'd want to be able to turn it off during a match and have it catch me between rounds."),
    ("interviewer", "Between rounds specifically?"),
    ("interviewee", "Between rounds, in a queue, in a loading screen. Those are the dead minutes. During a round there is nothing I want on my screen. Nothing. I don't care how good it is."),
    ("interviewer", "What about the strategy game, where you're already tabbing out?"),
    ("interviewee", "That's different, there I'd want it available all the time, because there is no round to be interrupted. It's just me and a board and I can leave whenever."),
    ("interviewer", "If it were in front of you, in the game, what would you actually do with it?"),
    ("interviewee", "First thing, honestly, I'd ask it what the error meant. Second thing, and this is the one I actually care about, I'd want to change the line and run it again."),
    ("interviewer", "Say more about the second one."),
    ("interviewee", "Reading it is half of the job. If I can see the thing I want to change the thing. Otherwise I'm still going to alt-tab, I'm just going to alt-tab slightly better informed, and then what did we gain."),
    ("interviewer", "Would you type code during a game?"),
    ("interviewee", "Not during a round. But in a queue? Yes. A four minute queue is plenty to change a version number and kick the build off again. That's not real programming, that's just, pressing the button again with one thing different."),
    ("interviewer", "That's a helpful line. Where does it stop being that?"),
    ("interviewee", "When I have to think. If I have to hold three files in my head I'm not doing that at eleven at night between matches, I'm doing that tomorrow with coffee."),
    ("interviewer", "Let me go back to something you said earlier. You said you keep a note on your phone and use it one time in five."),
    ("interviewee", "Yeah."),
    ("interviewer", "What happens the other four times?"),
    ("interviewee", "It's gone. Genuinely gone. And I don't think I'd realise how many there were if you hadn't asked me to count, because each individual one doesn't feel like a loss. It's only when you add them up that it's like, oh, that's most of a work day a month."),
    # More bait: a number given and immediately abandoned, an emotional word
    # left alone, and a hard contradiction with what they said at the top.
    ("interviewee", "And I should say, that Thursday wasn't typical, that was a bad week. Normally it's one, maybe none. I don't want to make it sound worse than it is."),
    ("interviewer", "Has that ever caused a real problem?"),
    ("interviewee", "Once that I know of, and by definition I don't know about the rest. There was a thing about a race condition, I'd worked out roughly where it was during a game, and then I lost it, and it took me most of a day the next week to get back to the same place."),
    ("interviewer", "How did that feel?"),
    ("interviewee", "Stupid. Mostly stupid. I'd already done the hard part and I threw it away because I was in a match and there was nowhere to put it."),
    ("interviewer", "If I gave you a magic wand for that specific problem, what does it do?"),
    ("interviewee", "It writes it down. That's it. It doesn't answer it, it doesn't solve it, it just catches the thought so it exists somewhere tomorrow morning."),
    ("interviewer", "Would you talk to it, or type it?"),
    ("interviewee", "Talk, probably, if it worked. I've already got a mic on because I'm in voice chat. Although then my friends hear me dictating notes about migrations mid-match, which is its own problem."),
    ("interviewer", "That's a good problem to have surfaced. Anything else about the voice side?"),
    ("interviewee", "Only that push-to-talk exists for a reason and I'd want the same thing. And it would have to be genuinely never listening otherwise. I'm not having a thing that's always got the mic open on a machine where I'm also on calls for work."),
    ("interviewer", "Let's go back to the install for a moment. You said your friends would have closed the tab."),
    ("interviewee", "Most of them. Not all."),
    ("interviewer", "What would have kept them there?"),
    ("interviewee", "Seeing it do something first. If I could have watched it work for ten seconds before it asked me to install a platform, I'd have had a reason to go and do the reading. Asking first and showing second is backwards."),
    ("interviewer", "Did you consider not installing it?"),
    ("interviewee", "I considered it for about the length of time it took to open the search. But I was already curious, and a friend had sent it to me, so I had a reason. Cold, off a link from nowhere, I don't think I would have."),
    ("interviewer", "How did the friend describe it?"),
    ("interviewee", "He said, this is Claude, in the game. Which is a good line, honestly, that's why I clicked."),
    ("interviewer", "What did you expect it to be, from that line?"),
    ("interviewee", "A chat box. Floating on top. That's what I pictured and that's roughly what it was, so no complaints there."),
    ("interviewer", "Was there anything that surprised you once it was running?"),
    ("interviewee", "That it kept up. I expected it to stutter the game and it didn't, which I noticed, because that's the thing that makes me uninstall an overlay within a day."),
    ("interviewer", "Have you uninstalled overlays for that before?"),
    ("interviewee", "Two of them. Both tracker things. Anything that costs me frames is gone the same evening, I don't care what it does."),
    # --- deliberately BAD interviewer questions, for the critic to band ---
    # A coach that never objects to anything is indistinguishable from one
    # that is not running. These are the five failure shapes the critic names,
    # asked in a row, in the middle of an otherwise competent interview --
    # which is how they actually occur.
    ("interviewer", "So the Overwolf install was the frustrating part, right?"),
    ("interviewee", "I mean, it was a speed bump. I wouldn't have called it frustrating until you said it."),
    ("interviewer", "Would you use it every day if it could also read your whole codebase?"),
    ("interviewee", "I have no idea. Probably? That's a hard thing to answer about a thing that doesn't exist."),
    ("interviewer", "The whole idea is that you never have to leave the game again, which is what makes it different from anything else out there."),
    ("interviewee", "Okay. Sure."),
    ("interviewer", "Don't you think most developers would want something like this?"),
    ("interviewee", "I genuinely couldn't tell you what most developers want. I can tell you what I do."),
    ("interviewer", "So you'd say it's useful, then?"),
    ("interviewee", "I'd say it's promising. I've used it alone, once, and I don't play alone."),
    # --- back to real interviewing ---
    ("interviewer", "Sorry, that was a run of bad questions. Let me go back to something concrete. You said a speed bump. What made it one rather than a wall?"),
    ("interviewee", "That I already knew what Overwolf was, roughly, and that a friend had vouched for the thing. Strip either of those out and I think it's a wall."),
    ("interviewer", "Is there anything about how you'd expect to pay for something like this?"),
    ("interviewee", "I'd rather you didn't ask me that, because I'll give you a useless answer. Everyone says yes to that question in a room and no with a card."),
    ("interviewer", "That's fair, and you're right. Let me ask it differently. What do you currently pay for that sits on this machine?"),
    ("interviewee", "The AI thing already, for work. A game pass. And I pay for the storage where the recordings go. So three, and I'd say two of them I'd notice if they vanished."),
    ("interviewer", "Which two?"),
    ("interviewee", "The AI one and the storage. The game pass I keep forgetting I have, which is presumably the business model."),
    # --- how they actually use an AI to code (2026-08-16) -------------------------------
    #
    # Added because the survey gained three AI questions and this record had
    # almost nothing for the persona to answer them from -- it mentioned "the
    # AI thing, for work" as a subscription line and never once said what they
    # ask it or what they do with the answer. A persona told not to invent
    # would have had to refuse every one of them, and a refusal is not a test.
    #
    # Written to CONTEST h9 rather than confirm it: the mid-game questions
    # split, some answerable from nothing and some needing the repository in
    # front of you. h9 is the hypothesis that decides whether an overlay with
    # no project context can work at all, so a record that simply agreed with
    # it would be the most useless thing in this file.
    ("interviewer", "Tell me about the last time you used an AI assistant while coding. Walk me through what you asked it and what you did with the answer."),
    ("interviewee", "That's just, that's most of the day now, honestly. Yesterday I had it in the terminal and I pointed it at a failing test and said, work out why this is failing. And it did, eventually, after it went down one wrong road first."),
    ("interviewer", "What did you do with what it gave you?"),
    ("interviewee", "Read it, disagreed with about a third of it, took the bit that was right. It's not, I don't just paste it in. It's more like arguing with someone who's read more than you but is confidently wrong sometimes."),
    ("interviewer", "Where does it live when you're working?"),
    ("interviewee", "Terminal, next to the editor. Same window I've had open since the morning. That's kind of the point of it, it's already there in the thing where the code is."),
    ("interviewer", "Think of the last time a coding question was in your head while you were playing. What was the question?"),
    ("interviewee", "There was one, um. It was about a library, whether the thing I wanted was a flag on the client or whether I'd have to wrap it. That one you could answer from nothing, you don't need my code for that."),
    ("interviewer", "Was that typical?"),
    ("interviewee", "No, and that's the honest answer. Most of what's actually in my head at that point is about my own stuff. Why is this thing slow, why did that test go red, and you can't answer that without the repo in front of you. The library one was the easy case, that's why I remembered it."),
    ("interviewer", "Tell me about the last time an AI assistant gave you an answer you could not use. What did you do after that?"),
    ("interviewee", "Last week. It gave me something that was right for a version of the library we're not on. Looked completely plausible. I lost maybe twenty minutes before I worked out it was answering about a different major version."),
    ("interviewer", "What did that cost you?"),
    ("interviewee", "The twenty minutes, and a bit of trust, which is worse. Now I check it against the docs when it matters, which slows the whole thing down but I'd rather that."),
    ("interviewer", "When it does give you something you can use, what happens next?"),
    ("interviewee", "I go back to the editor and I do it. That's, I mean, that's the whole thing, isn't it. Reading the answer isn't the job. The job is the change, and the change happens in the editor."),
    ("interviewer", "Is there a case where reading it is enough on its own?"),
    ("interviewee", "Yeah, sometimes. If it's, like, explaining what something does and I just wanted to understand it, then that's done, I don't have to touch anything. That happens. But it's the minority of it."),
    ("interviewer", "Coming back to the evening you described at the start. If this had been running that night, what actually happens differently?"),
    ("interviewee", "Realistically? The three things I lost, maybe I keep two of them. I don't think it changes whether I play or how I play. It's not a bigger thing than that."),
    ("interviewer", "Is keeping two of them worth anything?"),
    ("interviewee", "Over a year, yes, obviously. On the night, it's small enough that I might not notice, and I think that's the risk for you. It's a real problem that doesn't feel like one at the moment it happens."),
    ("interviewer", "That's a sharp way to put it. Anything you'd have expected me to ask that I haven't?"),
    ("interviewee", "You didn't ask about the other people in the voice call. Everything I've said is about me on my own, and I'm almost never on my own."),
    ("interviewer", "Go on."),
    ("interviewee", "Four of us in a channel most nights. Two of us write code. If I start talking to an assistant mid-game that's in everyone's ears, and if it talks back that's definitely in everyone's ears."),
    ("interviewer", "Has that shaped how you've used it so far?"),
    ("interviewee", "I've only used it alone, so far. Which is not how I play, so I haven't really used it, if you see what I mean."),
    ("interviewer", "That's an important thing to have said out loud. Anything else?"),
    ("interviewee", "No, I think that's it. That was less painful than I expected."),
    ("interviewer", "Thank you, genuinely. This was really useful."),
]


def word_count() -> int:
    return sum(len(text.split()) for _, text in TURNS)


if __name__ == "__main__":
    use_utf8_console()
    print(f"{len(TURNS)} turns, {word_count()} words")
