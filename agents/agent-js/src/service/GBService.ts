import type { ActionType, GameState } from "../types/GameTypes";

const host = "http://localhost";
const port = 8080;

const config = {
  "headless": false, 
  "sound": false,
  "load_autosave": false
}

const gbService = {
  async startGame(): Promise<GameState> {
    const response = await fetch(`${host}:${port}/initialize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(config)
    });
    return response.json();
  },

  async getGameState(): Promise<GameState> {
    const response = await fetch(`${host}:${port}/game_state`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json"
      }
    });
    return response.json();
  },

  async sendAction(action: ActionType): Promise<GameState> {
    const response = await fetch(`${host}:${port}/action`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(action)
    });
    return response.json();
  }
};

export default gbService;