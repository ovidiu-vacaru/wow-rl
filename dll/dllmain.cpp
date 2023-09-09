#include "pch.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <array>
#include <iostream>
#include <thread>
#include <string>
#include <windows.h>
#include <psapi.h>
#include <iostream>
#include <tlhelp32.h>

#pragma comment(lib, "ws2_32.lib") 


uintptr_t GetBaseAddress(DWORD processID, const wchar_t* moduleName)
{
    HANDLE processHandle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
    if (processHandle == NULL)
        return 0;

    HMODULE hMods[1024];
    DWORD cbNeeded;
    if (EnumProcessModules(processHandle, hMods, sizeof(hMods), &cbNeeded))
    {
        for (unsigned int i = 0; i < (cbNeeded / sizeof(HMODULE)); i++)
        {
            wchar_t szModName[MAX_PATH];
            if (GetModuleBaseNameW(processHandle, hMods[i], szModName, sizeof(szModName) / sizeof(wchar_t)))
            {
                if (wcscmp(szModName, moduleName) == 0)
                {
                    uintptr_t baseAddress = (uintptr_t)hMods[i];
                    CloseHandle(processHandle);
                    return baseAddress;
                }
            }
        }
    }

    CloseHandle(processHandle);
    return 0;
}

DWORD GetProcessID(const wchar_t* processName)
{
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnap == INVALID_HANDLE_VALUE)
    {
        return 0; // Snapshot failed
    }

    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);

    if (!Process32First(hSnap, &pe32))
    {
        CloseHandle(hSnap);
        return 0; // Failed to gather information
    }

    do
    {
        if (wcscmp(processName, pe32.szExeFile) == 0)
        {
            CloseHandle(hSnap);
            return pe32.th32ProcessID; // Found the process, return its PID
        }
    } while (Process32Next(hSnap, &pe32));

    CloseHandle(hSnap);
    return 0; // Process not found
}


class Session 
{
public:
    Session(SOCKET client_socket) : client_socket_(client_socket) {}

    void start() { do_read(); }

private:

    void do_read()
    {

        char buffer[1024];
        int iResult = recv(client_socket_, buffer, sizeof(buffer), 0);
        if (iResult > 0)
        {   
            DWORD processID = GetProcessID(L"Wow.exe");

            uintptr_t targetAddress1 = GetBaseAddress(processID, L"Wow.exe") + 0x6DF4E4;
            float value1 = *reinterpret_cast<float*>(targetAddress1);

            uintptr_t targetAddress2 = GetBaseAddress(processID, L"Wow.exe") + 0x6DF4E8;
            float value2 = *reinterpret_cast<float*>(targetAddress2);

            uintptr_t targetAddress3 = GetBaseAddress(processID, L"Wow.exe") + 0x6DF4EC;
            float value3 = *reinterpret_cast<float*>(targetAddress3);

            char buffer[sizeof(float) * 3];
            std::memcpy(&buffer[0], &value1, sizeof(value1));
            std::memcpy(&buffer[sizeof(float)], &value2, sizeof(value2));
            std::memcpy(&buffer[sizeof(float) * 2], &value3, sizeof(value3));

            do_write(buffer, sizeof(buffer));

        }
        else if (iResult == 0)
        {
            // Connection closing
            closesocket(client_socket_);
        }
        else
        {
            // An error occurred


            // Show a MessageBox with the message
         
            std::cerr << "recv failed: " << WSAGetLastError() << '\n';

            char message[128];
            sprintf_s(message, sizeof(message), "The error is: %d", WSAGetLastError());
            MessageBoxA(NULL, message, "Alert", MB_OK | MB_ICONINFORMATION);
            closesocket(client_socket_);
        }

    }
    void do_write(const char* data, size_t size)
    {
        int iSendResult = send(client_socket_, data, size, 0);
        if (iSendResult == SOCKET_ERROR)
        {
            closesocket(client_socket_);
        }
    }

    SOCKET client_socket_;
    std::array<char, 1024> data_;
};

class Server
{
public:
    Server(short port)
        : port_(port)
    {
        do_accept();
    }
private:
    void do_accept()
    {
        struct addrinfo* result = NULL, hints;
        ZeroMemory(&hints, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        hints.ai_flags = AI_PASSIVE;

        int iResult = getaddrinfo(NULL, std::to_string(port_).c_str(), &hints, &result);
        if (iResult != 0)
        {
            std::cerr << "getaddrinfo failed: " << iResult << '\n';
            WSACleanup();
            return;
        }

        SOCKET ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (ListenSocket == INVALID_SOCKET)
        {
            std::cerr << "Error at socket: " << WSAGetLastError() << '\n';
            freeaddrinfo(result);
            WSACleanup();
            return;
        }

        iResult = bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
        if (iResult == SOCKET_ERROR)
        {
            std::cerr << "bind failed with error: " << WSAGetLastError() << '\n';
            freeaddrinfo(result);
            closesocket(ListenSocket);
            WSACleanup();
            return;
        }

        freeaddrinfo(result);

        if (listen(ListenSocket, SOMAXCONN) == SOCKET_ERROR)
        {
            std::cerr << "Listen failed with error: " << WSAGetLastError() << '\n';
            closesocket(ListenSocket);
            WSACleanup();
            return;
        }

        SOCKET ClientSocket;
        while (true)
        {
            ClientSocket = accept(ListenSocket, NULL, NULL);
            if (ClientSocket == INVALID_SOCKET)
            {
                MessageBoxA(NULL, "meow", "INVALID", MB_OK | MB_ICONINFORMATION);
                std::cerr << "accept failed: " << WSAGetLastError() << '\n';
                closesocket(ListenSocket);
                WSACleanup();
                return;
            }
            std::unique_ptr<Session> session = std::make_unique<Session>(ClientSocket);
            session->start();
        }
    }

    short port_;
};

std::unique_ptr<Server> server;

// Thread function to start server
void StartServer(short port)
{
    server = std::make_unique<Server>(port);
}

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    {
        MessageBoxA(NULL, "meow", "Alert", MB_OK | MB_ICONINFORMATION);
        WSADATA wsaData; // this is a structure in winsock that contains information about  about the windows socket implementation WSAStartup
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
        {
            std::cerr << "WSAStartup failed: " << WSAGetLastError() << '\n';
            return FALSE;
        }

        short port = 12345; // Change this to your desired port
        std::thread t(StartServer, port);
        t.detach(); // Detach the thread

        break;
    }
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:

        if (lpReserved == NULL)
        {
            MessageBoxA(NULL, "meow", "Alert", MB_OK | MB_ICONINFORMATION);
        }
        else
        {
            MessageBoxA(NULL, "meow2", "Alert", MB_OK | MB_ICONINFORMATION);
        }
        
    break;

       
    }
    return TRUE;
}
