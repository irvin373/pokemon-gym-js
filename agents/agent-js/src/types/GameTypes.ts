export interface GameState {
  player_name: string;
  rival_name: string;
  money: number;
  location: string;
  coordinates: [number, number];
  badges: string[];
  valid_moves: string[];
  inventory: string[];
  dialog: string;
  pokemons: string[];
  screenshot_base64: string;
  collision_map: string | null;
  step_number: number;
  execution_time: number;
  score: number;
}

export type KeyType = 'up' | 'down' | 'left' | 'right' | 'a' | 'b' | 'start' | 'select';

export interface ActionType {
  action_type: string;
  keys?: KeyType[];
}
