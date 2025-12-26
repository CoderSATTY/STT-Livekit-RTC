import express from 'express';
import { AccessToken } from 'livekit-server-sdk';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(cors());

// Note: Added 'async' here
app.get('/token', async (req, res) => {
    try {
        const roomName = "my-room";
        const participantName = "user-" + Math.random().toString(36).substr(2, 6);

        const at = new AccessToken(
            process.env.LIVEKIT_API_KEY,
            process.env.LIVEKIT_API_SECRET,
            { identity: participantName }
        );

        at.addGrant({
            roomJoin: true,
            room: roomName,
            canPublish: true,
            canSubscribe: true,
        });

        // FIX: Added 'await' because toJwt() is now async in newer versions
        const token = await at.toJwt();

        console.log("✅ Generated Token:", token.substring(0, 15) + "...");
        res.json({ token: token });

    } catch (error) {
        console.error("❌ Error generating token:", error);
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => console.log("🚀 Token server running on port 3000"));