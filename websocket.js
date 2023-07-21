const { Neurosity } = require("@neurosity/sdk");
const ws = require('ws');
const dotenv = require("dotenv");
dotenv.config();
const wss = new ws.Server({ port: 8080 });

main(); 

async function main() {

  const neurosity = new Neurosity();

  await neurosity.login({
    email: process.env.NEUROSITY_EMAIL, 
    password: process.env.NEUROSITY_PASSWORD
  });

  console.log("Neurosity login successful");
  console.log("Listening for brainwaves on port 8080");
  neurosity.brainwaves("powerByBand").subscribe((brainwaves) => {
    // Send brainwave data to all connected websocket clients
    wss.clients.forEach(client => {
      if (client.readyState === ws.OPEN) {
        client.send(JSON.stringify(brainwaves));
      }
    });

  });

}
