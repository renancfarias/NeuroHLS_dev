#ifndef _DEBUG_FUNCTIONS_HPP_
#define _DEBUG_FUNCTIONS_HPP_

#include <ap_fixed.h>
#include <ap_int.h>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

template<int ch, int lines, int cols, typename mat_type>
void print_mat(mat_type mat[ch][lines][cols], string name)
{
    ofstream dnet("debug_net_VITIS.txt", ios::app);

    dnet << " *** " << name << ":" << endl;

    for (int c = 0; c < ch; c++)
    {
        dnet << "Channel " << c + 1 << ":" << endl;

        for (int i = 0; i < lines; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                dnet << fixed << setprecision(2) <<  mat[c][i][j] << " ";
            }

            dnet << endl;
        }

        if (c < ch - 1)
        {
            dnet << "--------------" << endl;
        }
        else
        {
            dnet << endl;
        }   
    }

    dnet.close();
}

template<int vet_size, typename vet_type>
void print_vet(vet_type v[vet_size], string name)
{
    ofstream dnet("debug_net_VITIS.txt", ios::app);

    dnet << " *** " << name << ":" << endl;

    for (int i = 0; i < vet_size; i++)
    {
        dnet << v[i] << " ";
    }

    dnet << endl << endl;

    dnet.close();
}

#endif // _DEBUG_FUNCTIONS_HPP_