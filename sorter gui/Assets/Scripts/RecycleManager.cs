using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;
using TMPro;
public class RecycleManager : MonoBehaviour
{
    static Socket listener;
    private CancellationTokenSource source;
    public ManualResetEvent allDone;

    public static readonly int PORT = 1755;
    public static readonly int WAITTIME = 1;

    public GameObject orgPrefab;   // Organic and recycling prefabs
    public GameObject recPrefab;
    private int objectToSpawn = -1;

    private int organicCount = 0; // Text for keeping track of waste counts
    private int recycleCount = 0;
    private int totalCount = 0;

    public TextMeshProUGUI organicText; // Text for updating the text in game
    public TextMeshProUGUI recycleText;
    public TextMeshProUGUI totalText;


    // Start is called before the first frame update
    RecycleManager(){
        source = new CancellationTokenSource(); // Initialize the CancellationTokenSource
        allDone = new ManualResetEvent(false);
    }
    async void Start()
    {
        await Task.Run(() => ListenEvents(source.Token));
    }
    void Update()
{
    if (objectToSpawn >= 0) // Every loop, check for active flag
    {
        
        GameObject prefabToSpawn = null;
        Vector3 spawnPosition = new Vector3(0f, 0f, 0f);
        Quaternion spawnRotation = Quaternion.identity; // Add random rotation to spawned object
        switch (objectToSpawn)
        {
            case 0: // Spawn the object, increment the count
                spawnPosition = new Vector3(50f, 15f, 25f);
                prefabToSpawn = orgPrefab;
                organicCount++;
                organicText.text = organicCount.ToString();
                break;
            case 1:
                spawnPosition = new Vector3(30f, 15f, 25f);
                prefabToSpawn = recPrefab;
                recycleCount++;
                recycleText.text = recycleCount.ToString();
                break;
            default:
                break;
        }

        if (prefabToSpawn != null) // Instantiate the game object
        {
            totalCount = organicCount + recycleCount;
            totalText.text = totalCount.ToString();
            spawnRotation = Quaternion.Euler(UnityEngine.Random.Range(0f, 360f), UnityEngine.Random.Range(0f, 360f), UnityEngine.Random.Range(0f, 360f));
            Instantiate(prefabToSpawn, spawnPosition, spawnRotation);
        }

        objectToSpawn = -1; // Reset the flag
    }
}

    private void ListenEvents(CancellationToken token) // Provided code from ENSC482 onedrive
    {

        IPHostEntry ipHostInfo = Dns.GetHostEntry(Dns.GetHostName());
        IPAddress ipAddress = ipHostInfo.AddressList.FirstOrDefault(ip => ip.AddressFamily == AddressFamily.InterNetwork);
        IPEndPoint localEndPoint = new IPEndPoint(ipAddress, PORT);

        listener = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

        try
        {
            listener.Bind(localEndPoint);
            listener.Listen(10);

            while (!token.IsCancellationRequested)
            {
                allDone.Reset();

                Debug.Log("Waiting for a connection... host: " + ipAddress.MapToIPv4().ToString() + " port: " + PORT);
                listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);

                while (!token.IsCancellationRequested)
                {
                    if (allDone.WaitOne(WAITTIME))
                    {
                        break;
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError(e.ToString());
        }
    }

    private void AcceptCallback(IAsyncResult ar)
    {
        Socket listener = (Socket)ar.AsyncState;
        Socket handler = listener.EndAccept(ar);

        allDone.Set();

        StateObject state = new StateObject();
        state.workSocket = handler;
        handler.BeginReceive(state.buffer, 0, StateObject.BufferSize, 0, new AsyncCallback(ReadCallback), state);
    }

    private void ReadCallback(IAsyncResult ar)
    {
        StateObject state = (StateObject)ar.AsyncState;
        Socket handler = state.workSocket;

        int read = handler.EndReceive(ar);

        if (read > 0)
        {
            state.recycling.Append(Encoding.ASCII.GetString(state.buffer, 0, read)); // need to fix???
            handler.BeginReceive(state.buffer, 0, StateObject.BufferSize, 0, new AsyncCallback(ReadCallback), state);
        }
        else
        {
            if (state.recycling.Length > 1) // need to fix????
            {
                string content = state.recycling.ToString();
                Debug.Log($"Read {content.Length} bytes from socket.\n Data : {content}");
                SpawnGameObject(content);
            }
            handler.Close();
        }
    }

    private void SpawnGameObject(string objectType) // This function is called upon receving data
{
    switch (objectType)
    {
        case "ORGA":
            objectToSpawn = 0;
            break;
        case "RECY":
            objectToSpawn = 1;
            break;
        default:
            Debug.LogWarning("Unknown object type received: " + objectType);
            objectToSpawn = -1; // Reset to invalid value
            break;
    }
}




    private void OnDestroy()
    {
        source.Cancel();
    }

    public class StateObject
    {
        public Socket workSocket = null;
        public const int BufferSize = 1024;
        public byte[] buffer = new byte[BufferSize];
        public StringBuilder recycling = new StringBuilder();
    };
}
